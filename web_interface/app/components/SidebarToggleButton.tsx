'use client';

interface SidebarToggleButtonProps {
  isOpen: boolean;
  onClick: () => void;
}

const SidebarToggleButton = ({ isOpen, onClick }: SidebarToggleButtonProps) => (
  <button 
    onClick={onClick}
    tabIndex={0}
    aria-label={isOpen ? "Close backend panel" : "Open backend panel"}
    style={{
      position: "fixed",
      right: isOpen ? "400px" : "0",
      top: "50%",
      transform: "translateY(-50%)",
      backgroundColor: isOpen ? "#EF4444" : "#4F46E5",
      color: "white",
      width: "36px",
      height: "100px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      borderTopLeftRadius: "8px",
      borderBottomLeftRadius: "8px",
      border: "none",
      cursor: "pointer",
      boxShadow: isOpen ? 
        "-4px 0 15px rgba(239, 68, 68, 0.3)" : 
        "-4px 0 15px rgba(79, 70, 229, 0.3)",
      zIndex: 9999,
      transition: "right 0.3s ease-in-out, background-color 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
      outline: "none"
    }}
    onFocus={(e) => {
      e.currentTarget.style.boxShadow = isOpen ? 
        "-4px 0 20px rgba(239, 68, 68, 0.5)" : 
        "-4px 0 20px rgba(79, 70, 229, 0.5)";
    }}
    onBlur={(e) => {
      e.currentTarget.style.boxShadow = isOpen ? 
        "-4px 0 15px rgba(239, 68, 68, 0.3)" : 
        "-4px 0 15px rgba(79, 70, 229, 0.3)";
    }}
    onMouseOver={(e) => {
      e.currentTarget.style.boxShadow = isOpen ? 
        "-4px 0 20px rgba(239, 68, 68, 0.5)" : 
        "-4px 0 20px rgba(79, 70, 229, 0.5)";
    }}
    onMouseOut={(e) => {
      e.currentTarget.style.boxShadow = isOpen ? 
        "-4px 0 15px rgba(239, 68, 68, 0.3)" : 
        "-4px 0 15px rgba(79, 70, 229, 0.3)";
    }}
  >
    <span style={{
      writingMode: "vertical-rl", 
      transform: "rotate(180deg)", 
      fontSize: "0.75rem",
      fontWeight: "bold",
      letterSpacing: "0.05em"
    }}>
      {isOpen ? "CLOSE" : "BACKEND"}
    </span>
  </button>
);

export default SidebarToggleButton; 