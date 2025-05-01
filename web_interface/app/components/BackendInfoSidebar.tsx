'use client';

import { createPortal } from 'react-dom';
import { useEffect, useState } from 'react';

// Get API base URL from environment or use default
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3030/api';

interface BackendInfoSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  activeTab: string;
}

const BackendInfoSidebar = ({ isOpen, onClose, activeTab }: BackendInfoSidebarProps) => {
  const [isMounted, setIsMounted] = useState(false);
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  // Only mount the portal on the client side
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Check API connection when sidebar is opened
  useEffect(() => {
    if (isOpen) {
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
            // Short timeout to avoid long waits if API is down
            signal: AbortSignal.timeout(2000)
          });
          
          setApiConnected(response.ok);
        } catch (error) {
          console.error("API connection check failed:", error);
          setApiConnected(false);
        }
      };
      
      checkApiConnection();
    }
  }, [isOpen]);

  // Define model information based on the active tab
  const getModelInfo = () => {
    switch (activeTab) {
      case "historical":
        return {
          models: [
            { name: "gaia_network/models/historical_data.py", description: "Processes historical climate and financial data" },
            { name: "demo/climate_risk_node.py", description: "Climate Risk Model (Node B)" }
          ],
          dataFiles: [
            { name: "data/historical/sea_level_1990_2020.csv", description: "Historical sea level measurements" },
            { name: "data/historical/property_values_1990_2020.csv", description: "Historical property values" },
            { name: "data/historical/insurance_premiums_1990_2020.csv", description: "Historical insurance premium index" },
            { name: "data/historical/storm_intensity_index_1990_2020.csv", description: "Historical storm intensity measurements" }
          ]
        };
      case "risks":
        return {
          models: [
            { name: "demo/real_estate_finance_node.py", description: "Real Estate Finance Model (Node A)" },
            { name: "demo/climate_risk_node.py", description: "Climate Risk Model (Node B)" },
            { name: "demo/actuarial_data_node.py", description: "Actuarial Data Model (Node C)" },
            { name: "gaia_network/distribution.py", description: "Statistical distribution processing" }
          ],
          dataFiles: [
            { name: "data/climate/ipcc_scenarios.json", description: "IPCC climate scenarios" },
            { name: "data/finance/risk_premium_tables.csv", description: "Risk premium data tables" }
          ]
        };
      case "scenarios":
        return {
          models: [
            { name: "demo/resilience_bond_node.py", description: "Resilience Bond Model (Node D)" },
            { name: "demo/real_estate_finance_node.py", description: "Real Estate Finance Model (Node A)" },
            { name: "gaia_network/query.py", description: "Query processing between models" }
          ],
          dataFiles: [
            { name: "data/scenarios/predefined_scenarios.json", description: "Pre-defined scenarios" },
            { name: "data/bond/resilience_metrics.csv", description: "Resilience bond metrics" }
          ]
        };
      case "simulation":
        return {
          models: [
            { name: "web_interface/app/api.py:run_simulation()", description: "Main simulation engine" },
            { name: "gaia_network/state.py", description: "State management for simulation" },
            { name: "demo/sfe_calculator.py", description: "System Free Energy calculations" }
          ],
          dataFiles: [
            { name: "data/climate/rcp_scenarios.json", description: "RCP climate scenarios" },
            { name: "data/adaptation/adaptation_measures.csv", description: "Climate adaptation measures" }
          ]
        };
      default:
        return {
          models: [
            { name: "gaia_network/node.py", description: "Base node implementation" }
          ],
          dataFiles: [
            { name: "data/general/general_parameters.json", description: "General simulation parameters" }
          ]
        };
    }
  };

  const modelInfo = getModelInfo();

  // For server-side rendering compatibility
  if (!isMounted) {
    return null;
  }

  // Add an overlay when the sidebar is open
  const overlayContent = isOpen ? (
    <div 
      onClick={onClose}
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.3)",
        zIndex: 9997,
        transition: "opacity 0.3s ease-in-out",
        opacity: isOpen ? 1 : 0,
        pointerEvents: isOpen ? "auto" : "none",
      }}
    />
  ) : null;

  // Update the sidebar styling for better visibility
  const sidebarContent = (
    <div style={{
      position: "fixed", 
      right: isOpen ? "0" : "-400px",
      top: "0",
      width: "400px",
      height: "100%",
      backgroundColor: "#FFFFFF",
      boxShadow: "-2px 0 10px rgba(0, 0, 0, 0.1)",
      zIndex: 9998,
      transition: "right 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
      padding: "1.5rem",
      overflow: "auto",
      animation: isOpen ? "sidebar-slide-in 0.3s ease-out forwards" : "none"
    }}>
      <style dangerouslySetInnerHTML={{
        __html: `
          @keyframes sidebar-slide-in {
            0% { transform: translateX(20px); opacity: 0.7; }
            100% { transform: translateX(0); opacity: 1; }
          }
        `
      }} />
      
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "1.5rem"
      }}>
        <h2 style={{
          fontSize: "1.25rem", 
          fontWeight: "bold",
          color: "#4338CA"
        }}>
          Backend Details
        </h2>
        <button 
          onClick={onClose}
          style={{
            backgroundColor: "transparent",
            border: "none",
            fontSize: "1.5rem",
            cursor: "pointer",
            color: "#64748B"
          }}
        >
          ×
        </button>
      </div>

      {/* Add API connection status indicator */}
      <div style={{
        padding: "0.75rem",
        borderRadius: "0.375rem",
        backgroundColor: apiConnected === null 
          ? "#F1F5F9" 
          : apiConnected 
            ? "#ECFDF5" 
            : "#FEF2F2",
        marginBottom: "1.5rem",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between"
      }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem"
        }}>
          <div style={{
            width: "0.75rem",
            height: "0.75rem",
            borderRadius: "50%",
            backgroundColor: apiConnected === null 
              ? "#94A3B8"
              : apiConnected 
                ? "#10B981" 
                : "#F87171"
          }}></div>
          <span style={{
            fontSize: "0.875rem",
            fontWeight: "600", 
            color: apiConnected === null 
              ? "#64748B" 
              : apiConnected 
                ? "#059669" 
                : "#B91C1C"
          }}>
            {apiConnected === null 
              ? "Checking API connection..." 
              : apiConnected 
                ? "Connected to Python API" 
                : "Using simulated data"}
          </span>
        </div>
        {!apiConnected && apiConnected !== null && (
          <div style={{
            fontSize: "0.75rem",
            color: "#64748B"
          }}>
            Running in demo mode
          </div>
        )}
      </div>

      <div style={{marginBottom: "1.5rem"}}>
        <h3 style={{
          fontSize: "1rem", 
          fontWeight: "600", 
          color: "#4338CA", 
          marginBottom: "0.75rem",
          backgroundColor: "#EEF2FF",
          padding: "0.5rem 0.75rem",
          borderRadius: "0.375rem"
        }}>
          Python Models Running
        </h3>
        <div style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem"
        }}>
          {modelInfo.models.map((model, index) => (
            <div key={index} style={{
              padding: "0.75rem",
              borderRadius: "0.375rem",
              backgroundColor: "#F8FAFC",
              border: "1px solid #E2E8F0"
            }}>
              <div style={{
                fontSize: "0.875rem", 
                fontWeight: "600", 
                fontFamily: "monospace",
                color: "#334155",
                marginBottom: "0.25rem"
              }}>
                {model.name}
              </div>
              <div style={{
                fontSize: "0.75rem", 
                color: "#64748B"
              }}>
                {model.description}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h3 style={{
          fontSize: "1rem", 
          fontWeight: "600", 
          color: "#059669", 
          marginBottom: "0.75rem",
          backgroundColor: "#ECFDF5",
          padding: "0.5rem 0.75rem",
          borderRadius: "0.375rem"
        }}>
          Data Files
        </h3>
        <div style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem"
        }}>
          {modelInfo.dataFiles.map((file, index) => (
            <div key={index} style={{
              padding: "0.75rem",
              borderRadius: "0.375rem",
              backgroundColor: "#F8FAFC",
              border: "1px solid #E2E8F0"
            }}>
              <div style={{
                fontSize: "0.875rem", 
                fontWeight: "600", 
                fontFamily: "monospace",
                color: "#334155",
                marginBottom: "0.25rem"
              }}>
                <a 
                  href={`/api/data-file?path=${encodeURIComponent(file.name)}`} 
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    color: "#0891B2",
                    textDecoration: "none",
                    display: "inline-block",
                    cursor: "pointer",
                    transition: "color 0.2s ease",
                    borderBottom: "1px dashed #0891B2"
                  }}
                  onMouseOver={(e) => {
                    e.currentTarget.style.color = "#0E7490";
                    e.currentTarget.style.borderBottom = "1px solid #0E7490";
                  }}
                  onMouseOut={(e) => {
                    e.currentTarget.style.color = "#0891B2";
                    e.currentTarget.style.borderBottom = "1px dashed #0891B2";
                  }}
                >
                  {file.name}
                </a>
              </div>
              <div style={{
                fontSize: "0.75rem", 
                color: "#64748B"
              }}>
                {file.description}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // Return both the overlay and the sidebar through the portal
  return createPortal(
    <>
      {overlayContent}
      {sidebarContent}
    </>,
    document.body
  );
};

export default BackendInfoSidebar; 