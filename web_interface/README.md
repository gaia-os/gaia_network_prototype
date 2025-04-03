# Gaia Network Web Interface

This is a modern web interface for the Gaia Network demo, built with Next.js, shadcn/ui, and Tailwind CSS.

## Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- The Gaia Network demo nodes running (ports 8011, 8012, 8013)

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install Node.js dependencies:
```bash
npm install
```

## Running the Application

1. First, start the Gaia Network demo nodes:
```bash
cd ..
PYTHONPATH=$PYTHONPATH:. python demo/run_web_demo.py
```

2. In a new terminal, start the Flask backend:
```bash
cd web_interface
python app.py
```

3. In another terminal, start the Next.js development server:
```bash
cd web_interface
npm run dev
```

4. Open your browser and navigate to http://localhost:3000

## Features

- Modern, responsive UI with dark mode support
- Real-time interaction with Gaia Network nodes
- Tabbed interface for different node types
- Query interface for each node
- Results display with confidence scores

## Development

The project uses:
- Next.js for the frontend framework
- shadcn/ui for UI components
- Tailwind CSS for styling
- Flask for the backend API
- TypeScript for type safety

## Project Structure

```
web_interface/
├── app/                    # Next.js app directory
│   ├── page.tsx           # Main page component
│   ├── layout.tsx         # Root layout
│   └── globals.css        # Global styles
├── components/            # React components
│   └── ui/               # shadcn/ui components
├── lib/                   # Utility functions
├── app.py                # Flask backend
└── requirements.txt      # Python dependencies
``` 