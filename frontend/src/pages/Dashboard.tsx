import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer
} from 'recharts';
import { TrendingUp, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import { fraudAPI, DetectionResponse } from '../services/api';

const COLORS = ['#22c55e', '#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6'];

interface Stats {
  total_transactions: number;
  fraud_count: number;
  normal_count: number;
  fraud_rate: number;
  categories: number;
  states: number;
  avg_amount: number;
  max_amount: number;
  min_amount: number;
}

interface CategoryData {
  category: string;
  fraud_rate: number;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [recentDetections, setRecentDetections] = useState<DetectionResponse[]>([]);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const statsData = await fraudAPI.getStatistics();
      setStats(statsData as Stats);
      
      // Simulate recent detections
      const sampleTransactions = [
        { amt: 500000, category: 'shopping_net', merchant: 'Shopee', city: 'Hồ Chí Minh', state: 'HCM', lat: 10.8231, long: 106.6297, merch_lat: 10.8221, merch_long: 106.6287 },
        { amt: 50, category: 'food_dining', merchant: 'Highlands', city: 'Hà Nội', state: 'HN', lat: 21.0285, long: 105.8542, merch_lat: 21.0275, merch_long: 105.8532 },
        { amt: 5000000, category: 'misc_net', merchant: 'Unknown', city: 'Đà Nẵng', state: 'ĐN', lat: 16.0471, long: 108.2068, merch_lat: 10.8231, merch_long: 106.6297 },
      ];
      
      const results = await fraudAPI.detectBatch(sampleTransactions);
      setRecentDetections(results.results);
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  // Prepare chart data
  const pieData = stats ? [
    { name: 'Normal', value: stats.normal_count, color: '#22c55e' },
    { name: 'Fraud', value: stats.fraud_count, color: '#ef4444' },
  ] : [];

  const categoryData: CategoryData[] = stats ? [
    { category: 'shopping_net', fraud_rate: 0.15 },
    { category: 'misc_net', fraud_rate: 0.12 },
    { category: 'gas_transport', fraud_rate: 0.08 },
    { category: 'food_dining', fraud_rate: 0.05 },
    { category: 'health_fitness', fraud_rate: 0.04 },
  ] : [];

  const timelineData = [
    { time: '00:00', fraud_rate: 0.02 },
    { time: '04:00', fraud_rate: 0.08 },
    { time: '08:00', fraud_rate: 0.05 },
    { time: '12:00', fraud_rate: 0.06 },
    { time: '16:00', fraud_rate: 0.07 },
    { time: '20:00', fraud_rate: 0.09 },
    { time: '23:00', fraud_rate: 0.11 },
  ];

  return (
    <div className="space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">Credit Card Fraud Detection Overview</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Transactions</p>
              <p className="text-3xl font-bold text-gray-900 mt-2">
                {stats?.total_transactions.toLocaleString()}
              </p>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Activity className="h-6 w-6 text-blue-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Fraud Transactions</p>
              <p className="text-3xl font-bold text-red-600 mt-2">
                {stats?.fraud_count.toLocaleString()}
              </p>
            </div>
            <div className="p-3 bg-red-50 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-red-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Normal Transactions</p>
              <p className="text-3xl font-bold text-green-600 mt-2">
                {stats?.normal_count.toLocaleString()}
              </p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Fraud Rate</p>
              <p className="text-3xl font-bold text-orange-600 mt-2">
                {stats?.fraud_rate}%
              </p>
            </div>
            <div className="p-3 bg-orange-50 rounded-lg">
              <TrendingUp className="h-6 w-6 text-orange-600" />
            </div>
          </div>
        </div>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Fraud vs Normal</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Category Chart */}
        <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Fraud by Category</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" />
              <YAxis />
              <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Fraud Rate']} />
              <Bar dataKey="fraud_rate" fill="#3b82f6" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Timeline Chart */}
      <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Fraud Rate by Hour</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Fraud Rate']} />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="fraud_rate" 
              stroke="#ef4444" 
              strokeWidth={3}
              dot={{ fill: '#ef4444', r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Detections */}
      <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Detections</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="pb-3 font-semibold text-gray-700">Amount</th>
                <th className="pb-3 font-semibold text-gray-700">Category</th>
                <th className="pb-3 font-semibold text-gray-700">Prediction</th>
                <th className="pb-3 font-semibold text-gray-700">Score</th>
                <th className="pb-3 font-semibold text-gray-700">Model</th>
              </tr>
            </thead>
            <tbody>
              {recentDetections.map((det, idx) => (
                <tr key={idx} className="border-b border-gray-100">
                  <td className="py-3">{det.details?.amount || 'N/A'}</td>
                  <td className="py-3">{det.details?.category || 'N/A'}</td>
                  <td className="py-3">
                    <span className={`px-2 py-1 rounded-full text-sm font-medium ${
                      det.prediction === 'fraud' 
                        ? 'bg-red-100 text-red-700' 
                        : 'bg-green-100 text-green-700'
                    }`}>
                      {det.prediction === 'fraud' ? 'FRAUD' : 'NORMAL'}
                    </span>
                  </td>
                  <td className="py-3">{(det.anomaly_score * 100).toFixed(1)}%</td>
                  <td className="py-3 text-sm text-gray-600">{det.model}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
