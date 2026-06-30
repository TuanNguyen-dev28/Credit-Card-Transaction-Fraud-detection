# Frontend - Credit Card Fraud Detection System

React + TypeScript + TailwindCSS frontend for the fraud detection system.

## Features

- **Dashboard**: Overview of fraud statistics and visualizations
- **Fraud Detection**: Interactive form to detect fraud in transactions
- **Real-time Results**: Instant prediction with confidence scores
- **Model Comparison**: View performance of all anomaly detection models

## Tech Stack

- React 18 with TypeScript
- TailwindCSS for styling
- Recharts for data visualization
- Axios for API communication
- Lucide React for icons
- React Router for navigation

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000/api/v1
```

## Available Scripts

- `npm start` - Start development server on port 3000
- `npm run build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Project Structure

```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/          # Page components
│   │   ├── Dashboard.tsx
│   │   └── FraudDetection.tsx
│   ├── services/       # API services
│   │   └── api.ts
│   ├── charts/         # Chart components
│   ├── App.tsx         # Main app component
│   ├── index.tsx       # Entry point
│   └── index.css       # Global styles
├── package.json
├── tsconfig.json
├── tailwind.config.js
└── postcss.config.js
```

## Docker

The frontend is containerized and available at `http://localhost:3000` when using Docker Compose.

## Contributing

Follow the existing code style and patterns. Use TypeScript for all new components.
