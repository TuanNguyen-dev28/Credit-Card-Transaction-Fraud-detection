import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react';
import { fraudAPI, TransactionRequest, DetectionResponse } from '../services/api';

const FraudDetection: React.FC = () => {
  const [formData, setFormData] = useState<TransactionRequest>({
    amt: 0,
    category: 'shopping_net',
    merchant: '',
    city: '',
    state: 'HCM',
    lat: 10.8231,
    long: 106.6297,
    merch_lat: 10.8221,
    merch_long: 106.6287,
    trans_date_trans_time: new Date().toISOString(),
    gender: 'M',
    city_pop: 9000000,
  });

  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fraudAPI.detect(formData);
      setResult(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Detection failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Page Title */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Fraud Detection</h1>
        <p className="text-gray-600 mt-1">
          Enter transaction details to detect potential fraud
        </p>
      </div>

      {/* Detection Form */}
      <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-sm p-6 border border-gray-200 space-y-6">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <Shield className="h-5 w-5 mr-2 text-primary-600" />
          Transaction Details
        </h2>

        {/* Amount */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Amount (VND)
          </label>
          <input
            type="number"
            name="amt"
            value={formData.amt}
            onChange={handleInputChange}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            placeholder="Enter amount"
            required
          />
        </div>

        {/* Category & Merchant */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Category
            </label>
            <select
              name="category"
              value={formData.category}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="food_dining">Food & Dining</option>
              <option value="gas_transport">Gas & Transport</option>
              <option value="grocery_pos">Grocery (POS)</option>
              <option value="grocery_net">Grocery (Online)</option>
              <option value="health_fitness">Health & Fitness</option>
              <option value="home">Home</option>
              <option value="kids_pets">Kids & Pets</option>
              <option value="misc_net">Misc (Online)</option>
              <option value="misc_pos">Misc (POS)</option>
              <option value="personal_care">Personal Care</option>
              <option value="shopping_net">Shopping (Online)</option>
              <option value="shopping_pos">Shopping (POS)</option>
              <option value="travel">Travel</option>
              <option value="entertainment">Entertainment</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Merchant
            </label>
            <input
              type="text"
              name="merchant"
              value={formData.merchant}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Enter merchant name"
              required
            />
          </div>
        </div>

        {/* City & State */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              City
            </label>
            <input
              type="text"
              name="city"
              value={formData.city}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              placeholder="Enter city"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              State/Province
            </label>
            <select
              name="state"
              value={formData.state}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="HN">Hà Nội</option>
              <option value="HCM">TP. Hồ Chí Minh</option>
              <option value="ĐN">Đà Nẵng</option>
              <option value="HP">Hải Phòng</option>
              <option value="CT">Cần Thơ</option>
              <option value="KH">Nha Trang</option>
              <option value="TTh">Huế</option>
              <option value="LĐ">Đà Lạt</option>
              <option value="NA">Nghệ An</option>
              <option value="HD">Hải Dương</option>
            </select>
          </div>
        </div>

        {/* Location */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Latitude
            </label>
            <input
              type="number"
              step="0.0001"
              name="lat"
              value={formData.lat}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Longitude
            </label>
            <input
              type="number"
              step="0.0001"
              name="long"
              value={formData.long}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Merchant Lat
            </label>
            <input
              type="number"
              step="0.0001"
              name="merch_lat"
              value={formData.merch_lat}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Merchant Long
            </label>
            <input
              type="number"
              step="0.0001"
              name="merch_long"
              value={formData.merch_long}
              onChange={handleInputChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-primary-600 text-white py-3 px-4 rounded-lg font-semibold hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 mr-2 animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Shield className="h-5 w-5 mr-2" />
              Detect Fraud
            </>
          )}
        </button>
      </form>

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className={`rounded-xl shadow-sm p-6 border-2 ${
          result.prediction === 'fraud'
            ? 'bg-red-50 border-red-200'
            : 'bg-green-50 border-green-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {result.prediction === 'fraud' ? (
                <AlertTriangle className="h-12 w-12 text-red-600" />
              ) : (
                <CheckCircle className="h-12 w-12 text-green-600" />
              )}
              <div>
                <h3 className={`text-2xl font-bold ${
                  result.prediction === 'fraud' ? 'text-red-700' : 'text-green-700'
                }`}>
                  {result.prediction === 'fraud' ? 'FRAUD DETECTED' : 'NORMAL TRANSACTION'}
                </h3>
                <p className="text-gray-600 mt-1">
                  Model: <span className="font-semibold">{result.model}</span>
                </p>
              </div>
            </div>

            <div className="text-right">
              <p className="text-sm text-gray-600">Anomaly Score</p>
              <p className={`text-3xl font-bold ${
                result.anomaly_score > 0.7 ? 'text-red-600' : 'text-green-600'
              }`}>
                {(result.anomaly_score * 100).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Confidence Bar */}
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Confidence</span>
              <span>{(result.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  result.prediction === 'fraud' ? 'bg-red-600' : 'bg-green-600'
                }`}
                style={{ width: `${result.confidence * 100}%` }}
              />
            </div>
          </div>

          {/* Details */}
          {result.details && (
            <div className="mt-4 pt-4 border-t border-gray-200">
              <p className="text-sm text-gray-600">
                Fraud votes: <span className="font-semibold">{result.details.fraud_votes}</span> / {result.details.total_models}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                {Object.entries(result.details.votes).map(([model, vote]) => (
                  <span
                    key={model}
                    className={`px-2 py-1 rounded text-xs font-medium ${
                      vote === 'fraud'
                        ? 'bg-red-100 text-red-700'
                        : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {model}: {vote}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FraudDetection;
