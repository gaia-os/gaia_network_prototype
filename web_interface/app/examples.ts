export interface ExampleScenario {
  title: string;
  description: string;
  steps: {
    title: string;
    description: string;
    query: {
      node: 'a' | 'b' | 'c';
      variable: string;
      covariates: Record<string, any>;
    };
    expectedResult: {
      value: number;
      confidence: number;
    };
  }[];
}

export const exampleScenarios: ExampleScenario[] = [
  {
    title: "Miami Beach Luxury Condo Development",
    description: "Evaluate the financial viability and climate resilience of a new luxury condo development in Miami Beach, considering climate risks and insurance implications.",
    steps: [
      {
        title: "Initial Climate Risk Assessment",
        description: "Check projected sea level rise and storm intensity for Miami Beach over the next 50 years under different IPCC scenarios.",
        query: {
          node: 'b',
          variable: 'sea_level_rise',
          covariates: {
            location: 'Miami Beach',
            scenario: 'RCP8.5',
            time_horizon: 50
          }
        },
        expectedResult: {
          value: 1.2,
          confidence: 0.85
        }
      },
      {
        title: "Historical Flood Analysis",
        description: "Analyze historical flood data and insurance claims to understand past flood patterns and their financial impact.",
        query: {
          node: 'c',
          variable: 'historical_flood_data',
          covariates: {
            location: 'Miami Beach',
            year: 2023
          }
        },
        expectedResult: {
          value: 0.15,
          confidence: 0.95
        }
      },
      {
        title: "Financial Projection",
        description: "Calculate expected ROI considering climate risks and adaptation measures.",
        query: {
          node: 'a',
          variable: 'expected_roi',
          covariates: {
            property_type: 'luxury_condo',
            location: 'Miami Beach',
            adaptation_strategy: 'elevated_construction',
            climate_risk_factor: 0.8
          }
        },
        expectedResult: {
          value: 0.12,
          confidence: 0.75
        }
      }
    ]
  },
  {
    title: "Resilience Bond Investment",
    description: "Evaluate the potential for a resilience bond to fund climate adaptation measures in a coastal city.",
    steps: [
      {
        title: "Storm Intensity Projection",
        description: "Assess projected storm intensity to understand future climate risks.",
        query: {
          node: 'b',
          variable: 'storm_intensity',
          covariates: {
            location: 'Miami',
            scenario: 'RCP4.5',
            time_horizon: 30
          }
        },
        expectedResult: {
          value: 1.5,
          confidence: 0.82
        }
      },
      {
        title: "Insurance Claims Analysis",
        description: "Review historical insurance claims to understand past storm damage costs.",
        query: {
          node: 'c',
          variable: 'insurance_claims',
          covariates: {
            location: 'Miami',
            year: 2023
          }
        },
        expectedResult: {
          value: 2500000,
          confidence: 0.90
        }
      },
      {
        title: "Bond ROI Assessment",
        description: "Calculate the risk-adjusted return on investment for the resilience bond.",
        query: {
          node: 'a',
          variable: 'bond_adjusted_roi',
          covariates: {
            bond_type: 'resilience_bond',
            location: 'Miami',
            adaptation_measures: ['flood_protection', 'storm_surge_barriers'],
            climate_risk_factor: 0.7
          }
        },
        expectedResult: {
          value: 0.08,
          confidence: 0.78
        }
      }
    ]
  },
  {
    title: "Climate-Adaptive Real Estate Portfolio",
    description: "Optimize a real estate portfolio by incorporating climate risk data and adaptation strategies.",
    steps: [
      {
        title: "Flood Risk Assessment",
        description: "Evaluate flood probability for multiple properties in the portfolio.",
        query: {
          node: 'b',
          variable: 'flood_probability',
          covariates: {
            location: 'Miami Beach',
            scenario: 'RCP6.0',
            time_horizon: 25
          }
        },
        expectedResult: {
          value: 0.35,
          confidence: 0.88
        }
      },
      {
        title: "Historical Performance Analysis",
        description: "Review historical flood data to understand past flood patterns.",
        query: {
          node: 'c',
          variable: 'historical_flood_data',
          covariates: {
            location: 'Miami Beach',
            year: 2023
          }
        },
        expectedResult: {
          value: 0.18,
          confidence: 0.92
        }
      },
      {
        title: "Portfolio Risk-Adjusted Returns",
        description: "Calculate risk-adjusted returns considering climate risks and adaptation measures.",
        query: {
          node: 'a',
          variable: 'risk_adjusted_roi',
          covariates: {
            portfolio_type: 'mixed_use',
            locations: ['Miami Beach', 'Fort Lauderdale'],
            adaptation_strategies: ['elevated_construction', 'flood_protection'],
            climate_risk_factor: 0.75
          }
        },
        expectedResult: {
          value: 0.15,
          confidence: 0.80
        }
      }
    ]
  }
]; 