import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export interface TransactionRequest {
  amt: number;
  category: string;
  merchant: string;
  city: string;
  state: string;
  lat: number;
  long: number;
  merch_lat: number;
  merch_long: number;
  trans_date_trans_time?: string;
  gender?: 'M' | 'F';
  city_pop?: number;
}

export interface DetectionResponse {
  prediction: 'fraud' | 'normal';
  model: string;
  anomaly_score: number;
  confidence: number;
  details?: Record<string, any>;
  timestamp: string;
}

export const fraudAPI = {
  // Detect fraud for single transaction
  detect: async (transaction: TransactionRequest): Promise<DetectionResponse> => {
    const response = await api.post<DetectionResponse>('/detect', transaction);
    return response.data;
  },

  // Batch detection
  detectBatch: async (transactions: TransactionRequest[]): Promise<{
    total: number;
    fraud_count: number;
    normal_count: number;
    results: DetectionResponse[];
    processing_time_ms: number;
  }> => {
    const response = await api.post('/detect/batch', { transactions });
    return response.data;
  },

  // Get models info
  getModelsInfo: async (): Promise<{
    models: Array<{ name: string; type: string; status: string }>;
    best_model: string;
  }> => {
    const response = await api.get('/models');
    return response.data;
  },

  // Get statistics
  getStatistics: async (): Promise<Record<string, any>> => {
    const response = await api.get('/stats');
    return response.data;
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; timestamp: string }> => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api;
