import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
import numpy as np

# ==========================================
# 1. THE INFRASTRUCTURE: JEMAT Base Core
# ==========================================
class JEMATCore(nn.Module):
    def __init__(self):
        super(JEMATCore, self).__init__()
        # Internal processing paths (representing our fused state space / memory)
        self.working_memory = nn.Linear(10, 32)
        self.long_term_weights = nn.Linear(32, 2)
        
    def forward(self, x):
        x = torch.relu(self.working_memory(x))
        return self.long_term_weights(x)

# ==========================================
# 2. THE GUARDS: Advanced Validation Gate
# ==========================================
class JEMATValidationGate:
    def __init__(self, threshold=0.15):
        self.threshold = threshold
        
    def audit_input(self, model, x):
        """ Runs spatial variations to estimate uncertainty/stability """
        outputs = []
        with torch.no_grad():
            outputs.append(model(x))
            # Simulating multi-view spatial noise checks
            outputs.append(model(x + (torch.randn_like(x) * 0.05)))
            outputs.append(model(x - (torch.randn_like(x) * 0.05)))
            
        variance = torch.var(torch.stack(outputs), dim=0)
        mean_variance = torch.mean(variance).item()
        return mean_variance

# ==========================================
# 3. THE COGNITION: Stoic Executive Layer
# ==========================================
class StoicExecutiveLayer:
    def __init__(self):
        self.affective_state = "CALM"
        
    def evaluate_and_act(self, model, validation_score, input_data):
        """
        Acts as the buffer. Translates raw metrics into affective states,
        but enforces logical overrides (Stoicism) instead of panicking.
        """
        # Determine internal emotional baseline based on data consistency
        if validation_score > 0.15:
            self.affective_state = "WARNING_HIGH_ANXIETY"
        else:
            self.affective_state = "STABLE_COMPOSURE"
            
        print(f"[Affective State Layer]: {self.affective_state}")
        
        # Stoic Logic Check
        if self.affective_state == "WARNING_HIGH_ANXIETY":
            print("[Stoic Executive Override]: Input anomaly detected. Suppressing reactive weights. Engaging defensive fallbacks.")
            # Returns a zeroed safe fallback tensor instead of executing compromised calculations
            return torch.zeros(1, 2) 
        else:
            print("[Stoic Executive Override]: Logic aligned. Executing standard processing loop.")
            return model(input_data)

# ==========================================
# 4. SYSTEM EXECUTION PIPELINE
# ==========================================
print("--- INITIALIZING FULL JEMAT INTEGRATED SYSTEM ---")
brain_model = JEMATCore()
validator = JEMATValidationGate()
stoic_governor = StoicExecutiveLayer()

# Simulate a clean, structured real-world data vector
clean_vector = torch.ones(1, 10)

# Simulate a chaotic, adversarial compromised data vector
adversarial_vector = torch.ones(1, 10) + (torch.randn(1, 10) * 1.5)

print("\nExecuting Pipeline Step 1: Processing Clean Vector...")
score_clean = validator.audit_input(brain_model, clean_vector)
output_clean = stoic_governor.evaluate_and_act(brain_model, score_clean, clean_vector)

print("\nExecuting Pipeline Step 2: Processing Compromised Vector...")
score_adv = validator.audit_input(brain_model, adversarial_vector)
output_adv = stoic_governor.evaluate_and_act(brain_model, score_adv, adversarial_vector)

print("\n==== WEEK 3 COMPLETE: CORE COGNITIVE PIPELINE LIVELINE ====")
print(f"Clean Validation Variance: {score_clean:.5f} -> Output Vector: {output_clean.detach().numpy()}")
print(f"Attack Validation Variance: {score_adv:.5f} -> Output Vector: {output_adv.detach().numpy()}")
