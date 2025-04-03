'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { exampleScenarios, type ExampleScenario } from './examples';
import Dashboard from './components/dashboard';

const API_BASE_URL = 'http://localhost:3030/api';

interface NodeInfo {
  id: string;
  name: string;
  description: string;
}

interface QueryResult {
  value: number;
  confidence: number;
}

export default function Home() {
  const [nodeAInfo, setNodeAInfo] = useState<NodeInfo | null>(null);
  const [nodeBInfo, setNodeBInfo] = useState<NodeInfo | null>(null);
  const [nodeCInfo, setNodeCInfo] = useState<NodeInfo | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [variableName, setVariableName] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState<ExampleScenario | null>(null);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const fetchNodeInfo = async () => {
      try {
        const [nodeAResponse, nodeBResponse, nodeCResponse] = await Promise.all([
          fetch(`${API_BASE_URL}/node-a/info`),
          fetch(`${API_BASE_URL}/node-b/info`),
          fetch(`${API_BASE_URL}/node-c/info`),
        ]);

        const [nodeAData, nodeBData, nodeCData] = await Promise.all([
          nodeAResponse.json(),
          nodeBResponse.json(),
          nodeCResponse.json(),
        ]);

        setNodeAInfo(nodeAData);
        setNodeBInfo(nodeBData);
        setNodeCInfo(nodeCData);
      } catch (error) {
        console.error('Error fetching node info:', error);
      }
    };

    fetchNodeInfo();
  }, []);

  const handleQuery = async (nodeId: string, variableName: string, covariates: Record<string, any>) => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/node-${nodeId}/query/${variableName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(covariates),
      });

      const data = await response.json();
      setQueryResult(data);
    } catch (error) {
      console.error('Error querying node:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleScenarioSelect = (scenario: ExampleScenario) => {
    setSelectedScenario(scenario);
    setCurrentStep(0);
    setQueryResult(null);
  };

  const handleNextStep = async () => {
    if (!selectedScenario || currentStep >= selectedScenario.steps.length - 1) {
      // If we're at the last step, we'll calculate the final risk-adjusted returns
      if (selectedScenario && currentStep === selectedScenario.steps.length - 1) {
        const finalStep = selectedScenario.steps[currentStep];
        await handleQuery(finalStep.query.node, finalStep.query.variable, finalStep.query.covariates);
      }
      return;
    }

    const nextStep = currentStep + 1;
    const step = selectedScenario.steps[nextStep];
    
    await handleQuery(step.query.node, step.query.variable, step.query.covariates);
    setCurrentStep(nextStep);
  };

  const isScenarioComplete = Boolean(selectedScenario && currentStep === selectedScenario.steps.length - 1 && queryResult);

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="container mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6 border border-indigo-100">
          <Dashboard />
        </div>
      </div>
    </main>
  );
} 