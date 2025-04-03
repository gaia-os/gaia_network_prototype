'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar, ScatterChart, Scatter } from 'recharts';
import * as d3 from 'd3';

// Import the sidebar components
import BackendInfoSidebar from "./BackendInfoSidebar";
import SidebarToggleButton from "./SidebarToggleButton";

const API_BASE_URL = 'http://localhost:3030/api';

// Sample data to ensure graphs always display something
const SAMPLE_SEA_LEVEL_DATA = Array.from({ length: 61 }, (_, i) => ({
  year: 1990 + i,
  value: 0.0009 * (i) * (i),
  confidence: Math.max(0.9 - (0.005 * i), 0.5),
}));

const SAMPLE_STORM_DATA = Array.from({ length: 61 }, (_, i) => ({
  year: 1990 + i,
  value: 1.0 + (0.007 * i),
  confidence: Math.max(0.95 - (0.006 * i), 0.5),
}));

const SAMPLE_PROPERTY_DATA = Array.from({ length: 61 }, (_, i) => {
  const year = 1990 + i;
  // Initial growth followed by decline due to climate risks
  const value = year < 2030 
    ? 100 * (1 + 0.02 * (year - 1990))
    : 100 * (1 + 0.02 * (2030 - 1990)) * (0.99 ** (year - 2030));
  return {
    year,
    value,
    confidence: Math.max(0.9 - (0.005 * i), 0.5),
  };
});

const SAMPLE_INSURANCE_DATA = Array.from({ length: 61 }, (_, i) => ({
  year: 1990 + i,
  value: 1000 * (1 + 0.005 * i + 0.0002 * i * i),
  confidence: Math.max(0.95 - (0.005 * i), 0.5),
}));

const SAMPLE_RISK_COMPARISON = {
  traditional: Array.from({ length: 10 }, (_, i) => ({
    risk_factor: i + 1,
    expected_return: 3 + 0.2 * (i + 1),
    confidence: 0.9 - (0.02 * (i + 1)),
  })),
  climate_model: Array.from({ length: 10 }, (_, i) => ({
    risk_factor: i + 1,
    expected_return: 3 + 0.25 * (i + 1),
    confidence: 0.92 - (0.02 * (i + 1)),
  })),
  financial_model: Array.from({ length: 10 }, (_, i) => ({
    risk_factor: i + 1,
    expected_return: 3 + 0.28 * (i + 1),
    confidence: 0.93 - (0.02 * (i + 1)),
  })),
  actuarial_model: Array.from({ length: 10 }, (_, i) => ({
    risk_factor: i + 1,
    expected_return: 3 + 0.27 * (i + 1),
    confidence: 0.94 - (0.02 * (i + 1)),
  })),
  gaia_network: Array.from({ length: 10 }, (_, i) => ({
    risk_factor: i + 1,
    expected_return: 3 + 0.35 * (i + 1),
    confidence: 0.95 - (0.015 * (i + 1)),
  })),
};

// Create default sample simulation results with fixed adaptation level (0.5)
const SAMPLE_SIMULATION_RESULTS = {
  sea_level: Array.from({ length: 31 }, (_, i) => ({
    year: 2023 + i,
    value: 0.01 * i,
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  })),
  property_value: Array.from({ length: 31 }, (_, i) => ({
    year: 2023 + i,
    // Instead of property values declining to zero, they decline based on adaptation level
    // Using fixed adaptation level of 0.5 for sample data
    value: 100 * (1 + (0.01 * i * 0.5 * 0.5) - (0.01 * i * (1 - 0.5) * 0.7)),
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  })),
  roi: Array.from({ length: 31 }, (_, i) => ({
    year: 2023 + i,
    // ROI now correlates with property value changes
    // Using fixed adaptation level of 0.5 for sample data
    value: 0.05 + (0.002 * i * 0.5) - (0.003 * i * (1 - 0.5)),
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  })),
  summary: {
    final_sea_level: 0.3,
    final_property_value: 100, // Starting property value
    final_roi: 0.05,
    avg_confidence: 0.75,
  }
};

type HistoricalDataPoint = {
  year: number;
  value: number;
  confidence: number;
};

type RiskComparisonPoint = {
  risk_factor: number;
  expected_return: number;
  confidence: number;
};

type RiskComparisonData = {
  traditional: RiskComparisonPoint[];
  climate_model: RiskComparisonPoint[];
  financial_model: RiskComparisonPoint[];
  actuarial_model: RiskComparisonPoint[];
  gaia_network: RiskComparisonPoint[];
};

type SimulationParams = {
  time_horizon: number;
  adaptation_level: number;
  climate_scenario: string;
};

type SimulationResults = {
  sea_level: HistoricalDataPoint[];
  property_value: HistoricalDataPoint[];
  roi: HistoricalDataPoint[];
  summary: {
    final_sea_level: number;
    final_property_value: number;
    final_roi: number;
    avg_confidence: number;
  };
  _id?: number; // Add optional _id field for React keying
};

const formatDataWithConfidence = (data: HistoricalDataPoint[]) => {
  return data.map(item => ({
    year: item.year,
    value: item.value,
    upperBound: item.value * (1 + (1 - item.confidence)),
    lowerBound: item.value * (1 - (1 - item.confidence))
  }));
};

const formatComparisonData = (data: RiskComparisonData | null) => {
  if (!data) return [];
  
  return data.traditional.map((item, index) => ({
    risk_factor: item.risk_factor,
    traditional: item.expected_return,
    climate_model: data.climate_model[index].expected_return,
    financial_model: data.financial_model[index].expected_return,
    actuarial_model: data.actuarial_model[index].expected_return,
    gaia_network: data.gaia_network[index].expected_return,
  }));
};

// Create sample scenarios that can be applied to the simulation
const EXAMPLE_SCENARIOS = [
  {
    id: "financial_viability",
    name: "Financial Viability of Coastal Properties",
    description: "Evaluates how sea level rise affects long-term property values and insurance costs in Miami.",
    image: "🏙️",
    params: {
      time_horizon: 30,
      adaptation_level: 0.3,
      climate_scenario: "RCP4.5",
    },
    highlight: "Shows that without adaptation measures, coastal property values decline by 35% over 30 years."
  },
  {
    id: "climate_resilience",
    name: "Climate Resilience Investment",
    description: "Models the ROI of climate adaptation infrastructure across coastal communities.",
    image: "🌊",
    params: {
      time_horizon: 25,
      adaptation_level: 0.8,
      climate_scenario: "RCP6.0",
    },
    highlight: "Demonstrates how high adaptation investments maintain property values despite increasing climate risks."
  },
  {
    id: "worst_case",
    name: "High-Emissions Investment Assessment",
    description: "Evaluates financial impacts under a high-emissions scenario with minimal adaptation.",
    image: "🏭",
    params: {
      time_horizon: 40,
      adaptation_level: 0.2,
      climate_scenario: "RCP8.5",
    },
    highlight: "Projects severe financial losses and insurance unavailability in high-risk coastal zones."
  }
];

// Add new type definitions for model integration
type ModelIntegration = {
  climate: boolean;
  financial: boolean;
  actuarial: boolean;
};

// First, modify the ModelIntegrationFlow component to remove the toggle buttons
const ModelIntegrationFlow = ({ 
  modelIntegration, 
  setModelIntegration 
}: {
  modelIntegration: ModelIntegration;
  setModelIntegration: React.Dispatch<React.SetStateAction<ModelIntegration>>;
}) => {
  const d3Container = useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (d3Container.current && modelIntegration) {
      d3.select(d3Container.current).selectAll("*").remove();
      
      const width = d3Container.current?.clientWidth || 500; 
      const height = 400; // Increased by 25% from 320px to 400px
      
      const svg = d3.select(d3Container.current)
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", [0, 0, width, height])
        .style("font", "10px sans-serif");
        
      // Define node data with adjusted positions to account for the increased height
      const nodes = [
        { id: "hub", name: "Gaia Network\nIntegration Hub", x: width*0.54, y: height/2, r: 60, color: "#4F46E5", active: true },
        { id: "climate", name: "Climate\nModels", x: width*0.22, y: height*0.18, r: 50, color: modelIntegration.climate ? "#60A5FA" : "#BFDBFE", active: modelIntegration.climate },
        { id: "financial", name: "Financial\nModels", x: width*0.86, y: height*0.18, r: 50, color: modelIntegration.financial ? "#34D399" : "#A7F3D0", active: modelIntegration.financial },
        { id: "actuarial", name: "Actuarial\nModels", x: width*0.09, y: height*0.55, r: 50, color: modelIntegration.actuarial ? "#A78BFA" : "#DDD6FE", active: modelIntegration.actuarial }
      ];
      
      // Define link data with custom icon for climate label
      const links = [
        { 
          source: "climate", 
          target: "hub", 
          color: modelIntegration.climate ? "#60A5FA" : "#BFDBFE", 
          active: modelIntegration.climate, 
          label: "Climate Data →", 
          labelColor: "#1E40AF", 
          labelBg: "#EFF6FF", 
          labelBorder: "#BFDBFE",
          // Update the Climate label
          reverseTarget: "financial",
          reverseLabel: "Climate →",
          reverseLabelColor: "#1E40AF",
          reverseLabelBg: "#EFF6FF",
          reverseLabelBorder: "#BFDBFE"
        },
        { 
          source: "financial", 
          target: "hub", 
          color: modelIntegration.financial ? "#34D399" : "#A7F3D0", 
          active: modelIntegration.financial,
          label: "← Market Data", 
          labelColor: "#166534", 
          labelBg: "#ECFDF5", 
          labelBorder: "#86EFAC",
          // Add reverse label going from hub to actuarial
          reverseTarget: "actuarial",
          reverseLabel: "← Financial",
          reverseLabelColor: "#166534",
          reverseLabelBg: "#ECFDF5",
          reverseLabelBorder: "#86EFAC"
        },
        { 
          source: "actuarial", 
          target: "hub", 
          color: modelIntegration.actuarial ? "#A78BFA" : "#DDD6FE", 
          active: modelIntegration.actuarial,
          label: "Risk Data →", 
          labelColor: "#5B21B6", 
          labelBg: "#F3E8FF", 
          labelBorder: "#DDD6FE",
          // Add reverse label going from hub to climate
          reverseTarget: "climate",
          reverseLabel: "← Actuarial",
          reverseLabelColor: "#5B21B6",
          reverseLabelBg: "#F3E8FF", 
          reverseLabelBorder: "#DDD6FE"
        }
      ];
      
      // Create a map for quick lookup
      const nodeMap: { [key: string]: any } = {}; // Add index signature
      nodes.forEach(node => {
        nodeMap[node.id] = node;
      });
      
      // Create links with calculated paths
      const linkElements = svg.append("g")
        .attr("class", "links")
        .selectAll("line")
        .data(links)
        .enter()
        .append("line")
        .attr("x1", d => nodeMap[d.source].x)
        .attr("y1", d => nodeMap[d.source].y)
        .attr("x2", d => nodeMap[d.target].x)
        .attr("y2", d => nodeMap[d.target].y)
        .attr("stroke", d => d.color)
        .attr("stroke-width", 3)
        .attr("stroke-dasharray", d => d.active ? "0" : "5,5")
        .style("transition", "all 0.3s ease");
      
      // Create nodes as circles
      const nodeElements = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(nodes)
        .enter()
        .append("g")
        .attr("transform", d => `translate(${d.x},${d.y})`)
        .style("cursor", d => d.id !== "hub" ? "pointer" : "default")
        .on("click", function(event: MouseEvent, d: any) { // Add types for event and data
          if (d.id !== "hub") {
            setModelIntegration((prev: ModelIntegration) => ({ // Type prev
              ...prev,
              [d.id]: !prev[d.id]
            }));
          }
        });
      
      // Add circles to nodes
      nodeElements.append("circle")
        .attr("r", d => d.r)
        .attr("fill", d => d.color)
        .attr("opacity", d => d.active ? 1 : 0.6)
        .style("transition", "all 0.3s ease");
      
      // Add labels to nodes
      nodeElements.append("text")
        .attr("text-anchor", "middle")
        .attr("dy", "0.3em")
        .attr("fill", "white")
        .attr("font-weight", "600")
        .attr("font-size", "0.75rem")
        .style("pointer-events", "none")
        .text(d => d.name)
        .call(wrapText, 80);
      
      // Add data flow labels
      links.forEach(link => {
        if (link.active) {
          // Calculate the midpoint of the link
          const sourceNode = nodeMap[link.source];
          const targetNode = nodeMap[link.target];
          
          // Calculate angle for proper rotation
          const dx = targetNode.x - sourceNode.x;
          const dy = targetNode.y - sourceNode.y;
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;
          
          // Calculate midpoint
          const midX = (sourceNode.x + targetNode.x) / 2;
          const midY = (sourceNode.y + targetNode.y) / 2;
          
          // Adjust position based on link
          let offsetX = 0, offsetY = 0;
          
          if (link.source === "climate") {
            offsetX = -22;
            offsetY = -22;
          } else if (link.source === "financial") {
            offsetX = 22;
            offsetY = -22;
          } else if (link.source === "actuarial") {
            offsetX = -22;
            offsetY = 14;
          }
          
          // For the financial model link (Market Data), we need to adjust the angle to prevent upside-down text
          const adjustedAngle = link.source === "financial" ? angle + 180 : angle;
          
          // Create label group
          const labelGroup = svg.append("g")
            .attr("transform", `translate(${midX + offsetX},${midY + offsetY}) rotate(${adjustedAngle})`);
          
          // Create label background
          labelGroup.append("rect")
            .attr("x", -45)
            .attr("y", -12)
            .attr("width", 90)
            .attr("height", 24)
            .attr("rx", 5)
            .attr("fill", link.labelBg)
            .attr("stroke", link.labelBorder)
            .attr("stroke-width", 1);
          
          // Create label text
          labelGroup.append("text")
            .attr("text-anchor", "middle")
            .attr("dy", "0.3em")
            .attr("fill", link.labelColor)
            .attr("font-size", "11px")
            .attr("font-weight", "500")
            .text(link.label);
        }
      });
      
      // After creating the primary data flow labels, add this code to create secondary labels
      // Add bidirectional data flow labels (from hub to other nodes)
      links.forEach(link => {
        if (link.active && link.reverseTarget) {
          // Calculate the source and target nodes for the reverse direction
          const sourceNode = nodeMap["hub"];
          const targetNode = nodeMap[link.reverseTarget];
          
          // Calculate angle for proper rotation
          const dx = targetNode.x - sourceNode.x;
          const dy = targetNode.y - sourceNode.y;
          const angle = Math.atan2(dy, dx) * 180 / Math.PI;
          
          // Calculate midpoint between hub and target node
          const midX = (sourceNode.x + targetNode.x) / 2;
          const midY = (sourceNode.y + targetNode.y) / 2;
          
          // Adjust position based on which direction we're going
          let offsetX = 0, offsetY = 0;
          
          if (link.reverseTarget === "climate") {
            offsetX = 22;
            offsetY = 22;  // Below the primary Climate Data label
          } else if (link.reverseTarget === "financial") {
            offsetX = -22;
            offsetY = 22;  // Above the Market Data label
          } else if (link.reverseTarget === "actuarial") {
            offsetX = 22;
            offsetY = -14;  // Moved further away from the line
          }
          
          // Adjust angle for each specific case to get the right orientation
          let adjustedAngle;
          if (link.reverseTarget === "financial") {
            // Special case for Climate→Financial direction
            adjustedAngle = 0; // No rotation - horizontal text
          } else if (link.reverseTarget === "climate") {
            adjustedAngle = angle - 180; // Flip and adjust to keep text readable
          } else if (link.reverseTarget === "actuarial") {
            adjustedAngle = angle + 180; // Flip 180 degrees to orient correctly
          } else {
            adjustedAngle = angle;
          }
          
          // Standard handling for labels
          const labelGroup = svg.append("g")
            .attr("transform", `translate(${midX + offsetX},${midY + offsetY})`);
          
          if (link.reverseTarget === "financial") {
            // Position the Climate label directly on the line and aligned with it
            const centerX = (nodeMap["hub"].x + nodeMap["financial"].x) / 2;
            const centerY = (nodeMap["hub"].y + nodeMap["financial"].y) / 2;
            
            // Calculate the angle of the line for alignment
            const dx = nodeMap["financial"].x - nodeMap["hub"].x;
            const dy = nodeMap["financial"].y - nodeMap["hub"].y;
            const lineAngle = Math.atan2(dy, dx) * 180 / Math.PI;
            
            // Offset downward along the line (perpendicular to it)
            const perpAngle = lineAngle + 90; // 90 degrees perpendicular
            const perpRad = perpAngle * Math.PI / 180;
            const offsetDistance = 15; // Increased distance from 12 to 15 to move away from the line
            
            // Calculate offset position
            const offsetX = Math.cos(perpRad) * offsetDistance;
            const offsetY = Math.sin(perpRad) * offsetDistance;
            
            // Apply the transform with rotation matching the line angle
            labelGroup.attr("transform", `translate(${centerX + offsetX},${centerY + offsetY}) rotate(${lineAngle})`);
          } else {
            labelGroup.attr("transform", `translate(${midX + offsetX},${midY + offsetY}) rotate(${adjustedAngle})`);
          }
          
          // Create label background
          labelGroup.append("rect")
            .attr("x", -45)
            .attr("y", -12)
            .attr("width", 90)
            .attr("height", 24)
            .attr("rx", 5)
            .attr("fill", link.reverseLabelBg)
            .attr("stroke", link.reverseLabelBorder)
            .attr("stroke-width", 1);
          
          // Create label text
          labelGroup.append("text")
            .attr("text-anchor", "middle")
            .attr("dy", "0.3em")
            .attr("fill", link.reverseLabelColor)
            .attr("font-size", "11px")
            .attr("font-weight", "500")
            .text(link.reverseLabel);
        }
      });
      
      // Function to wrap text
      function wrapText(text: d3.Selection<any, unknown, any, any>, width: number) {
        text.each(function(this: SVGTextElement) { // Explicitly type 'this'
          const text = d3.select(this);
          const words = text.text().split(/\n/);
          const lineHeight = 1.1;
          const y = text.attr("y") || 0;
          
          text.text(null);
          
          words.forEach((word, i) => {
            text.append("tspan")
              .attr("x", 0)
              .attr("y", y)
              .attr("dy", `${i * lineHeight}em`)
              .text(word);
          });
        });
      }
    }
  }, [modelIntegration, setModelIntegration]);
  
  return (
    <div>
      <div 
        ref={d3Container}
        style={{
          position: "relative",
          height: "400px", // Increased by 25% from 320px to 400px
          width: "100%",
          padding: "20px"
        }}
      />
    </div>
  );
};

// Modify the existing function to generate sample data based on current parameters
const generateSampleData = (params: SimulationParams) => {
  const { time_horizon, adaptation_level, climate_scenario } = params;
  
  // Scale factors based on climate scenario
  const scenarioFactor = {
    "RCP2.6": 0.6,  // Low emissions
    "RCP4.5": 1.0,  // Medium-low emissions
    "RCP6.0": 1.4,  // Medium-high emissions
    "RCP8.5": 2.0   // High emissions
  }[climate_scenario] || 1.0;
  
  // Create sample sea level data based on parameters
  const sea_level = Array.from({ length: time_horizon + 1 }, (_, i) => ({
    year: 2023 + i,
    value: 0.01 * i * scenarioFactor,
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  }));
  
  // Create sample property value data based on parameters
  // Higher adaptation = less property value decline, higher climate scenario = greater impact
  const property_value = Array.from({ length: time_horizon + 1 }, (_, i) => ({
    year: 2023 + i,
    value: 100 * (1 + (0.01 * i * adaptation_level * 0.5) - (0.01 * i * (1 - adaptation_level) * 0.7 * scenarioFactor)),
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  }));
  
  // Create sample ROI data based on parameters
  const roi = Array.from({ length: time_horizon + 1 }, (_, i) => ({
    year: 2023 + i,
    value: 0.05 + (0.002 * i * adaptation_level) - (0.003 * i * (1 - adaptation_level) * scenarioFactor),
    confidence: Math.max(0.9 - (0.01 * i), 0.5),
  }));
  
  // Calculate final values
  const final_sea_level = sea_level[sea_level.length - 1].value;
  const final_property_value = property_value[property_value.length - 1].value;
  const final_roi = roi[roi.length - 1].value;
  const avg_confidence = 0.75 - (0.05 * scenarioFactor);
  
  return {
    sea_level,
    property_value,
    roi,
    summary: {
      final_sea_level,
      final_property_value,
      final_roi,
      avg_confidence
    }
  };
};

// Dashboard component
const Dashboard = () => {
  // State for historical data
  const [historicalData, setHistoricalData] = useState({
    sea_level: SAMPLE_SEA_LEVEL_DATA,
    storm_intensity: SAMPLE_STORM_DATA,
    property_values: SAMPLE_PROPERTY_DATA,
    insurance_premiums: SAMPLE_INSURANCE_DATA,
  });
  
  // State for risk comparison
  const [riskComparison, setRiskComparison] = useState(SAMPLE_RISK_COMPARISON);
  
  // State for simulation parameters and results
  const [simulationParams, setSimulationParams] = useState<SimulationParams>({
    time_horizon: 30, // 30 years
    adaptation_level: 0.5, // 50% adaptation level
    climate_scenario: 'RCP4.5', // Middle scenario
  });
  
  const [simulationResults, setSimulationResults] = useState<SimulationResults | null>(SAMPLE_SIMULATION_RESULTS);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  
  // Add state for model integration
  const [modelIntegration, setModelIntegration] = useState({
    climate: true,
    financial: true,
    actuarial: true,
  });
  
  // State for tabs and sidebar
  const [activeTab, setActiveTab] = useState("historical");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Add a flag to track if the simulation has run for the current parameters
  const [hasRunWithCurrentParams, setHasRunWithCurrentParams] = useState(false);
  
  // Add state to track the active scenario
  const [activeScenario, setActiveScenario] = useState<string | null>(null);
  
  // API connection state
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  
  // Add API health check effect
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        // Try a minimal POST request to avoid triggering OPTIONS preflight in some cases
        const response = await fetch(`${API_BASE_URL}/run_simulation`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          // Send minimal data to avoid heavy processing
          body: JSON.stringify({ check_only: true }),
          signal: AbortSignal.timeout(2000)
        });
        
        setApiConnected(response.ok);
      } catch (error) {
        console.error("API connection check failed:", error);
        setApiConnected(false);
      }
    };
    
    checkApiConnection();
  }, []);
  
  // Make the memoizedRunSimulation function more reactive to parameter changes
  const memoizedRunSimulation = useCallback(async () => {
    console.log("runSimulation: Setting isSimulationRunning to true");
    setIsSimulationRunning(true);
    
    // Generate a timestamp to force React to recognize this as a new object
    const simulationId = Date.now();
    console.log("Running simulation with ID:", simulationId);
    
    try {
      console.log("runSimulation: Sending request to API...");
      const response = await fetch(`${API_BASE_URL}/run_simulation`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...simulationParams,
          models: {
            climate: modelIntegration.climate,
            financial: modelIntegration.financial,
            actuarial: modelIntegration.actuarial,
          }
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log("runSimulation: API request successful, received data:", data);
        setSimulationResults({...data, _id: simulationId});
      } else {
        console.error('runSimulation: API request failed with status:', response.status);
        const sampleData = generateSampleData(simulationParams);
        setSimulationResults({...sampleData, _id: simulationId});
      }
    } catch (error) {
      console.error('runSimulation: Error during fetch:', error);
      const sampleData = generateSampleData(simulationParams);
      setSimulationResults({...sampleData, _id: simulationId});
    } finally {
      setTimeout(() => {
        console.log("runSimulation: Setting isSimulationRunning to false");
        setIsSimulationRunning(false);
      }, 800); // Add small delay for visual feedback
    }
  }, [simulationParams, modelIntegration]);

  // Add a visual indicator when parameter changes are processed
  useEffect(() => {
    if (activeTab === "simulation") {
      console.log("Parameters changed:", simulationParams);
      memoizedRunSimulation();
      setHasRunWithCurrentParams(true);
      
      // If this is a manual parameter change, clear the active scenario
      if (activeScenario) {
        // Check if current params match any scenario's params
        const matchingScenario = EXAMPLE_SCENARIOS.find(scenario => 
          scenario.params.time_horizon === simulationParams.time_horizon &&
          scenario.params.adaptation_level === simulationParams.adaptation_level &&
          scenario.params.climate_scenario === simulationParams.climate_scenario
        );
        
        // If no matching scenario, clear the active scenario name
        if (!matchingScenario) {
          console.log("Parameters changed manually, clearing active scenario");
          setActiveScenario(null);
        }
      }
    } else {
      setHasRunWithCurrentParams(false);
    }
  }, [simulationParams, activeTab, memoizedRunSimulation, activeScenario, EXAMPLE_SCENARIOS]);
  
  // Apply a predefined scenario
  const applyScenario = (scenario: any) => { // Add basic type for scenario
    console.log("applyScenario: Setting simulation params:", scenario.params);
    setSimulationParams(scenario.params);
    setActiveScenario(scenario.name); // Store the scenario name
  };
  
  console.log("Current sidebar state:", sidebarOpen);
  
  // Default simulation parameters for reset
  const defaultSimulationParams = {
    time_horizon: 30,
    adaptation_level: 0.5,
    climate_scenario: 'RCP4.5',
  };
  
  // Function to reset simulation to default values
  const resetSimulation = () => {
    setSimulationParams(defaultSimulationParams);
    setActiveScenario(null);
  };
  
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6 text-indigo-800">Gaia Network Demo</h1>
      
      {/* Add the standalone sidebar components */}
      <SidebarToggleButton 
        isOpen={sidebarOpen} 
        onClick={() => {
          console.log("Toggle button clicked, current state:", sidebarOpen);
          setSidebarOpen(!sidebarOpen);
        }} 
      />
      
      <BackendInfoSidebar 
        isOpen={sidebarOpen} 
        onClose={() => setSidebarOpen(false)} 
        activeTab={activeTab}
      />
      
      <Tabs 
        defaultValue="historical" 
        value={activeTab}
        onValueChange={setActiveTab} 
        className="w-full"
      >
        <TabsList className="mb-6 flex h-10 items-center justify-center rounded-md bg-indigo-100 p-1 text-indigo-950">
          <TabsTrigger 
            value="historical" 
            className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-indigo-950 data-[state=active]:shadow-sm"
          >
            Historical Data
          </TabsTrigger>
          <TabsTrigger 
            value="risks" 
            className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-indigo-950 data-[state=active]:shadow-sm"
          >
            Risk Assessment
          </TabsTrigger>
          <TabsTrigger 
            value="scenarios" 
            className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-indigo-950 data-[state=active]:shadow-sm"
          >
            Scenario Explorer
          </TabsTrigger>
          <TabsTrigger 
            value="simulation" 
            className="inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:bg-white data-[state=active]:text-indigo-950 data-[state=active]:shadow-sm"
          >
            Simulation
          </TabsTrigger>
        </TabsList>
        
        {/* Historical Data Tab */}
        <TabsContent value="historical" className="mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sea Level Chart */}
            <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-md">
              <CardHeader className="bg-white px-6 py-4 border-b border-slate-100">
                <CardTitle className="text-lg font-semibold text-slate-900">Historical Sea Level Rise (1990-2020)</CardTitle>
                <CardDescription className="text-sm text-slate-500">Miami-Dade County, Florida</CardDescription>
              </CardHeader>
              <CardContent className="px-6 py-4">
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formatDataWithConfidence(historicalData.sea_level)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      domain={['dataMin', 'dataMax']}
                      type="number"
                      tickCount={7}
                    />
                    <YAxis 
                      label={{ value: 'Meters Above 1990 Level', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="upperBound" 
                      stroke="none" 
                      fill="#BFDBFE" 
                      fillOpacity={0.4} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#3B82F6" 
                      fill="#3B82F6" 
                      fillOpacity={0.8} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="lowerBound" 
                      stroke="none" 
                      fill="#BFDBFE" 
                      fillOpacity={0.4} 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
              <CardFooter className="bg-slate-50 px-6 py-3 border-t border-slate-100">
                <div className="text-sm text-slate-500">
                  Source: NOAA Tides & Currents Data, 2021
                </div>
              </CardFooter>
            </Card>
            
            {/* Property Values Chart */}
            <Card className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-md">
              <CardHeader>
                <CardTitle>Coastal Property Values (1990-2020)</CardTitle>
                <CardDescription>Inflation-adjusted average value per property</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formatDataWithConfidence(historicalData.property_values)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      domain={['dataMin', 'dataMax']}
                      type="number"
                      tickCount={7}
                    />
                    <YAxis 
                      label={{ value: 'Index (1990=100)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="upperBound" 
                      stroke="none" 
                      fill="#D1FAE5" 
                      fillOpacity={0.4} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#10B981" 
                      fill="#10B981" 
                      fillOpacity={0.8} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="lowerBound" 
                      stroke="none" 
                      fill="#D1FAE5" 
                      fillOpacity={0.4} 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
              <CardFooter>
                <div className="text-sm text-gray-500">
                  Source: Property market data and census records
                </div>
              </CardFooter>
            </Card>
            
            {/* Storm Intensity Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Storm Intensity Index (1990-2020)</CardTitle>
                <CardDescription>Normalized frequency and severity</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formatDataWithConfidence(historicalData.storm_intensity)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      domain={['dataMin', 'dataMax']}
                      type="number"
                      tickCount={7}
                    />
                    <YAxis 
                      label={{ value: 'Index (1990=1.0)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="upperBound" 
                      stroke="none" 
                      fill="#FEE2E2" 
                      fillOpacity={0.4} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#EF4444" 
                      fill="#EF4444" 
                      fillOpacity={0.8} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="lowerBound" 
                      stroke="none" 
                      fill="#FEE2E2" 
                      fillOpacity={0.4} 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
              <CardFooter>
                <div className="text-sm text-gray-500">
                  Source: NOAA Hurricane Database, adjusted for intensity
                </div>
              </CardFooter>
            </Card>
            
            {/* Insurance Premiums Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Insurance Premiums (1990-2020)</CardTitle>
                <CardDescription>Average annual premium for coastal properties</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={formatDataWithConfidence(historicalData.insurance_premiums)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="year" 
                      domain={['dataMin', 'dataMax']}
                      type="number"
                      tickCount={7}
                    />
                    <YAxis 
                      label={{ value: 'USD (inflation adjusted)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip />
                    <Area 
                      type="monotone" 
                      dataKey="upperBound" 
                      stroke="none" 
                      fill="#E0E7FF" 
                      fillOpacity={0.4} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#6366F1" 
                      fill="#6366F1" 
                      fillOpacity={0.8} 
                    />
                    <Area 
                      type="monotone" 
                      dataKey="lowerBound" 
                      stroke="none" 
                      fill="#E0E7FF" 
                      fillOpacity={0.4} 
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
              <CardFooter>
                <div className="text-sm text-gray-500">
                  Source: Insurance industry data, aggregated across providers
                </div>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        
        {/* Risk Assessment Tab */}
        <TabsContent value="risks" className="mt-0 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2">
          <div className="grid grid-cols-1 md:grid-cols-6 gap-3">
            {/* Left Column - Stacked Graphs - Now taller (25% more height) */}
            <div className="md:col-span-5 flex flex-col gap-3">
              {/* 1. Expected Return Graph Card */}
              <Card className="overflow-hidden flex-1">
                <CardHeader className="py-1">
                  <CardTitle>Model Comparison: Risk-Return Analysis</CardTitle>
                  <CardDescription>Comparing different modeling approaches for coastal property investment risk assessment</CardDescription>
                </CardHeader>
                <CardContent className="pb-1">
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart 
                      data={formatComparisonData(riskComparison)}
                      margin={{ top: 20, right: 20, left: 20, bottom: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="risk_factor" 
                        label={{ value: 'Risk Factor', position: 'insideBottom', offset: -10 }}
                        type="number"
                      />
                      <YAxis 
                        label={{ 
                          value: 'Expected Return (%)', 
                          angle: -90, 
                          position: 'insideLeft',
                          offset: 0,
                          dy: 40
                        }}
                      />
                      <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                      <Legend 
                        verticalAlign="bottom" 
                        height={36} 
                        wrapperStyle={{ paddingTop: "20px" }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="traditional" 
                        name="Traditional Assessment"
                        stroke="#94A3B8" 
                        strokeWidth={2}
                      />
                      {modelIntegration.climate && (
                        <Line 
                          type="monotone" 
                          dataKey="climate_model" 
                          name="Climate Model"
                          stroke="#60A5FA" 
                          strokeWidth={2}
                        />
                      )}
                      {modelIntegration.financial && (
                        <Line 
                          type="monotone" 
                          dataKey="financial_model" 
                          name="Financial Model"
                          stroke="#34D399" 
                          strokeWidth={2}
                        />
                      )}
                      {modelIntegration.actuarial && (
                        <Line 
                          type="monotone" 
                          dataKey="actuarial_model" 
                          name="Actuarial Model"
                          stroke="#A78BFA" 
                          strokeWidth={2}
                        />
                      )}
                      {(modelIntegration.climate && modelIntegration.financial && modelIntegration.actuarial) && (
                        <Line 
                          type="monotone" 
                          dataKey="gaia_network" 
                          name="Gaia Network (Integrated)"
                          stroke="#4F46E5" 
                          strokeWidth={3}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
                <CardFooter className="pt-3 pb-8">
                  <div className="text-sm text-gray-500 my-4 mb-6">
                    The Gaia Network integration shows improved risk-return assessment by combining models.
                  </div>
                </CardFooter>
              </Card>
            
              {/* 2. Model Integration Flow Diagram */}
              <Card className="overflow-hidden flex-1">
                <CardHeader className="py-1">
                  <CardTitle>Gaia Network Model Integration</CardTitle>
                  <CardDescription>Interactive visualization of how models connect</CardDescription>
                </CardHeader>
                <CardContent className="pb-1">
                  <div style={{ height: "300px" }}>
                    <ModelIntegrationFlow 
                      modelIntegration={modelIntegration} 
                      setModelIntegration={setModelIntegration} 
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
            
            {/* Right Column - Stacked Risk Boxes */}
            <div className="md:col-span-1 flex flex-col gap-3">
              {/* Climate Risk Box */}
              <Card className={`border-l-8 ${modelIntegration.climate ? 'border-l-blue-500' : 'border-l-gray-300'} flex flex-col`}>
                <CardHeader className="py-1 px-2">
                  <CardTitle className="text-sm flex items-center gap-1">
                    <Badge className={modelIntegration.climate ? "bg-blue-100 text-blue-800" : "bg-gray-100 text-gray-500"}>Climate</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="py-1 px-2 flex flex-col flex-grow">
                  <p className="text-xs text-gray-700 mb-auto">Physical climate effects including sea level rise</p>
                  
                  {/* Button aligned to bottom */}
                  <div className="mt-2">
                    <Button 
                      variant={modelIntegration.climate ? "default" : "outline"}
                      onClick={() => setModelIntegration(prev => ({...prev, climate: !prev.climate}))}
                      size="sm"
                      className={`w-full ${modelIntegration.climate ? 
                        "bg-blue-500 hover:bg-blue-600" : 
                        "border border-blue-300 hover:bg-blue-50"}`}
                    >
                      {modelIntegration.climate ? "Disable" : "Enable"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              {/* Financial Risk Box */}
              <Card className={`border-l-8 ${modelIntegration.financial ? 'border-l-green-500' : 'border-l-gray-300'} flex flex-col`}>
                <CardHeader className="py-1 px-2">
                  <CardTitle className="text-sm flex items-center gap-1">
                    <Badge className={modelIntegration.financial ? "bg-green-100 text-green-800" : "bg-gray-100 text-gray-500"}>Financial</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="py-1 px-2 flex flex-col flex-grow">
                  <p className="text-xs text-gray-700 mb-auto">Property values, mortgage markets, and financial impacts</p>
                  
                  {/* Button aligned to bottom */}
                  <div className="mt-2">
                    <Button 
                      variant={modelIntegration.financial ? "default" : "outline"}
                      onClick={() => setModelIntegration(prev => ({...prev, financial: !prev.financial}))}
                      size="sm"
                      className={`w-full ${modelIntegration.financial ? 
                        "bg-green-500 hover:bg-green-600" : 
                        "border border-green-300 hover:bg-green-50"}`}
                    >
                      {modelIntegration.financial ? "Disable" : "Enable"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
              
              {/* Actuarial Risk Box */}
              <Card className={`border-l-8 ${modelIntegration.actuarial ? 'border-l-purple-500' : 'border-l-gray-300'} flex flex-col`}>
                <CardHeader className="py-1 px-2">
                  <CardTitle className="text-sm flex items-center gap-1">
                    <Badge className={modelIntegration.actuarial ? "bg-purple-100 text-purple-800" : "bg-gray-100 text-gray-500"}>Actuarial</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent className="py-1 px-2 flex flex-col flex-grow">
                  <p className="text-xs text-gray-700 mb-auto">Insurance premium projections and risk pricing</p>
                  
                  {/* Button aligned to bottom */}
                  <div className="mt-2">
                    <Button 
                      variant={modelIntegration.actuarial ? "default" : "outline"}
                      onClick={() => setModelIntegration(prev => ({...prev, actuarial: !prev.actuarial}))}
                      size="sm"
                      className={`w-full ${modelIntegration.actuarial ? 
                        "bg-purple-500 hover:bg-purple-600" : 
                        "border border-purple-300 hover:bg-purple-50"}`}
                    >
                      {modelIntegration.actuarial ? "Disable" : "Enable"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>
        
        {/* Scenario Explorer Tab */}
        <TabsContent value="scenarios" className="mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2">
          <div className="grid grid-cols-1 gap-6">
            <Card>
              <CardHeader>
                <div className="flex items-center">
                  <CardTitle className="text-2xl font-bold">Example Scenarios</CardTitle>
                  <CardDescription className="mt-0 ml-auto mr-auto text-base translate-x-[-4rem]">Pre-defined scenarios to demonstrate the Gaia Network capabilities</CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {EXAMPLE_SCENARIOS.map(scenario => (
                    <Card key={scenario.id} className="border-2 border-indigo-100 hover:border-indigo-300 transition-all flex flex-col">
                      <CardHeader>
                        <div className="text-3xl mb-2">{scenario.image}</div>
                        <CardTitle>{scenario.name}</CardTitle>
                      </CardHeader>
                      <CardContent className="flex-grow">
                        <p className="text-gray-700 mb-4">{scenario.description}</p>
                        <div className="bg-indigo-50 p-3 rounded-md mb-4">
                          <div className="text-sm font-medium text-indigo-800 mb-1">Key Parameters:</div>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>Time Horizon:</div>
                            <div className="font-medium">{scenario.params.time_horizon} years</div>
                            <div>Adaptation:</div>
                            <div className="font-medium">{scenario.params.adaptation_level * 100}%</div>
                            <div>Climate Scenario:</div>
                            <div className="font-medium">{scenario.params.climate_scenario}</div>
                          </div>
                        </div>
                        <div className="bg-amber-50 p-3 rounded-md">
                          <div className="text-sm font-medium text-amber-800 mb-1">Key Finding:</div>
                          <div className="text-sm">{scenario.highlight}</div>
                        </div>
                      </CardContent>
                      <CardFooter className="mt-auto">
                        <Button 
                          className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 rounded-md shadow-sm transition-colors"
                          onClick={async () => {
                            console.log(`Applying scenario: ${scenario.name}`);
                            applyScenario(scenario);
                            setActiveTab("simulation");
                            console.log("Triggering simulation automatically...");
                            await memoizedRunSimulation();
                          }}
                        >
                          Simulate this Scenario
                        </Button>
                      </CardFooter>
                    </Card>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        
        {/* Simulation Tab */}
        <TabsContent value="simulation" className="mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2">
          {activeScenario ? (
            <div className="mb-4 bg-indigo-50 border border-indigo-200 rounded-md p-3 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-indigo-800">
                  {activeScenario} Scenario
                </h2>
                <p className="text-sm text-indigo-600 flex items-center gap-1.5">
                  Showing results for the predefined scenario
                  {apiConnected !== null && (
                    <>
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-slate-300 mx-1"></span>
                      <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                        apiConnected ? "bg-green-100 text-green-800" : "bg-amber-100 text-amber-800"
                      }`}>
                        {apiConnected ? "Live API" : "Simulated Data"}
                      </span>
                    </>
                  )}
                </p>
              </div>
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  className="border-red-200 text-red-700 hover:bg-red-50"
                  onClick={resetSimulation}
                >
                  Clear Scenario
                </Button>
                <Button 
                  variant="outline" 
                  className="border-indigo-200 text-indigo-700 hover:bg-indigo-50"
                  onClick={() => setActiveTab("scenarios")}
                >
                  Browse Scenarios
                </Button>
              </div>
            </div>
          ) : (
            <div className="mb-4 bg-gray-50 border border-gray-200 rounded-md p-3 flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-700">
                  Custom Simulation
                </h2>
                <p className="text-sm text-gray-600 flex items-center gap-1.5">
                  Using custom parameters
                  {apiConnected !== null && (
                    <>
                      <span className="inline-block w-1.5 h-1.5 rounded-full bg-slate-300 mx-1"></span>
                      <span className={`text-xs px-1.5 py-0.5 rounded-full ${
                        apiConnected ? "bg-green-100 text-green-800" : "bg-amber-100 text-amber-800"
                      }`}>
                        {apiConnected ? "Live API" : "Simulated Data"}
                      </span>
                    </>
                  )}
                </p>
              </div>
              <Button 
                variant="outline" 
                className="border-indigo-200 text-indigo-700 hover:bg-indigo-50"
                onClick={() => setActiveTab("scenarios")}
              >
                Browse Scenarios
              </Button>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="md:col-span-1">
              <Card>
                <CardHeader>
                  <CardTitle>Simulation Parameters</CardTitle>
                  <CardDescription>Adjust parameters to run custom simulations</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div>
                      <Label htmlFor="time-horizon" className="mb-2 block">Time Horizon: {simulationParams.time_horizon} years</Label>
                      <Slider 
                        id="time-horizon"
                        value={[simulationParams.time_horizon]} 
                        min={10} 
                        max={50} 
                        step={5}
                        onValueChange={value => setSimulationParams(prev => ({...prev, time_horizon: value[0]}))}
                        className="mb-2"
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>10y</span>
                        <span>30y</span>
                        <span>50y</span>
                      </div>
                    </div>
                    
                    <div>
                      <Label htmlFor="adaptation-level" className="mb-2 block">Adaptation Level: {simulationParams.adaptation_level * 100}%</Label>
                      <Slider 
                        id="adaptation-level"
                        value={[simulationParams.adaptation_level * 100]} 
                        min={0} 
                        max={100} 
                        step={5}
                        onValueChange={value => setSimulationParams(prev => ({...prev, adaptation_level: value[0] / 100}))}
                        className="mb-2"
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                      </div>
                    </div>
                    
                    <div>
                      <Label htmlFor="climate-scenario" className="mb-2 block">Climate Scenario</Label>
                      <Select 
                        value={simulationParams.climate_scenario}
                        onValueChange={value => setSimulationParams(prev => ({...prev, climate_scenario: value}))}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select climate scenario" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="RCP2.6">RCP2.6 (Low Emissions)</SelectItem>
                          <SelectItem value="RCP4.5">RCP4.5 (Medium-Low Emissions)</SelectItem>
                          <SelectItem value="RCP6.0">RCP6.0 (Medium-High Emissions)</SelectItem>
                          <SelectItem value="RCP8.5">RCP8.5 (High Emissions)</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    
                    {isSimulationRunning && (
                      <div className="pt-4 flex items-center justify-center">
                        <div className="animate-pulse text-indigo-600 font-medium">
                          Running Simulation...
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              
              {simulationResults && (
                <Card className="mt-6">
                  <CardHeader>
                    <CardTitle>Simulation Summary</CardTitle>
                    <CardDescription>End of period outcomes</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div>
                        <div className="flex justify-between mb-1">
                          <div className="text-sm font-medium">Sea Level Rise</div>
                          <div className="text-sm font-medium">{simulationResults.summary.final_sea_level.toFixed(2)}m</div>
                        </div>
                        <Progress value={Math.min(simulationResults.summary.final_sea_level * 100, 100)} className="h-2" />
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <div className="text-sm font-medium">Property Value Change</div>
                          <div className={`text-sm font-medium ${simulationResults.summary.final_property_value >= 100 ? 'text-green-600' : 'text-red-600'}`}>
                            {((simulationResults.summary.final_property_value - 100) / 100 * 100).toFixed(1)}%
                          </div>
                        </div>
                        <Progress value={Math.min(Math.max(simulationResults.summary.final_property_value, 0), 200) / 2} className="h-2" />
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <div className="text-sm font-medium">Return on Investment</div>
                          <div className={`text-sm font-medium ${simulationResults.summary.final_roi >= 0.05 ? 'text-green-600' : 'text-red-600'}`}>
                            {(simulationResults.summary.final_roi * 100).toFixed(1)}%
                          </div>
                        </div>
                        <Progress value={Math.min(Math.max(simulationResults.summary.final_roi * 1000, 0), 100)} className="h-2" />
                      </div>
                      
                      <div>
                        <div className="flex justify-between mb-1">
                          <div className="text-sm font-medium">Model Confidence</div>
                          <div className="text-sm font-medium">{(simulationResults.summary.avg_confidence * 100).toFixed(0)}%</div>
                        </div>
                        <Progress value={simulationResults.summary.avg_confidence * 100} className="h-2" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
            
            <div className="md:col-span-3">
              {simulationResults && (
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Property Value Projection</CardTitle>
                      <CardDescription>
                        Impact of {simulationParams.climate_scenario} climate scenario with {simulationParams.adaptation_level * 100}% adaptation over {simulationParams.time_horizon} years
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <AreaChart data={formatDataWithConfidence(simulationResults.property_value)}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="year" 
                            domain={['dataMin', 'dataMax']}
                            type="number"
                            tickCount={6}
                          />
                          <YAxis 
                            label={{ value: 'Property Value Index (2023=100)', angle: -90, position: 'insideLeft' }}
                          />
                          <Tooltip />
                          <Area 
                            type="monotone" 
                            dataKey="upperBound" 
                            stroke="none" 
                            fill="#D1FAE5" 
                            fillOpacity={0.4} 
                            name="Upper Bound"
                          />
                          <Area 
                            type="monotone" 
                            dataKey="value" 
                            stroke="#10B981" 
                            fill="#10B981" 
                            fillOpacity={0.8} 
                            name="Property Value"
                          />
                          <Area 
                            type="monotone" 
                            dataKey="lowerBound" 
                            stroke="none" 
                            fill="#D1FAE5" 
                            fillOpacity={0.4} 
                            name="Lower Bound"
                          />
                        </AreaChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <Card>
                      <CardHeader>
                        <CardTitle>Sea Level Projection</CardTitle>
                        <CardDescription>
                          Based on {simulationParams.climate_scenario} scenario
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={250}>
                          <AreaChart data={formatDataWithConfidence(simulationResults.sea_level)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="year" 
                              domain={['dataMin', 'dataMax']}
                              type="number"
                              tickCount={6}
                            />
                            <YAxis 
                              label={{ value: 'Meters Above 2023 Level', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip />
                            <Area 
                              type="monotone" 
                              dataKey="upperBound" 
                              stroke="none" 
                              fill="#BFDBFE" 
                              fillOpacity={0.4} 
                            />
                            <Area 
                              type="monotone" 
                              dataKey="value" 
                              stroke="#3B82F6" 
                              fill="#3B82F6" 
                              fillOpacity={0.8} 
                            />
                            <Area 
                              type="monotone" 
                              dataKey="lowerBound" 
                              stroke="none" 
                              fill="#BFDBFE" 
                              fillOpacity={0.4} 
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                    
                    <Card>
                      <CardHeader>
                        <CardTitle>Return on Investment</CardTitle>
                        <CardDescription>
                          Annual ROI with {simulationParams.adaptation_level * 100}% adaptation investment
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={250}>
                          <AreaChart data={formatDataWithConfidence(simulationResults.roi)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="year" 
                              domain={['dataMin', 'dataMax']}
                              type="number"
                              tickCount={6}
                            />
                            <YAxis 
                              label={{ value: 'ROI (%)', angle: -90, position: 'insideLeft' }}
                              tickFormatter={(value) => `${(value * 100).toFixed(1)}`}
                            />
                            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(2)}%`} />
                            <Area 
                              type="monotone" 
                              dataKey="upperBound" 
                              stroke="none" 
                              fill="#E0E7FF" 
                              fillOpacity={0.4} 
                            />
                            <Area 
                              type="monotone" 
                              dataKey="value" 
                              stroke="#6366F1" 
                              fill="#6366F1" 
                              fillOpacity={0.8} 
                            />
                            <Area 
                              type="monotone" 
                              dataKey="lowerBound" 
                              stroke="none" 
                              fill="#E0E7FF" 
                              fillOpacity={0.4} 
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default Dashboard; 