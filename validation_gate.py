import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# 1. Load a tiny, pre-trained model as our base processing engine
class BaseEngine(nn.Module):
    def __init__(self):
        super(BaseEngine, self).__init__()
        # Using a lightweight ResNet18
        self.model = models.resnet18(pretrained=True)
        self.model.eval() # Set to evaluation mode
        
    def forward(self, x):
        return self.model(x)

# 2. Build Your Custom Layered Validation Gate
class ValidationGate:
    def __init__(self, threshold=1.5):
        self.threshold = threshold # Sensitivity to inconsistencies
        
    def multi_view_check(self, model, original_tensor):
        """
        Transforms the image slightly (multi-view) to see if the 
        model's logic remains temporally/spatially stable.
        """
        # Create a slightly altered view (e.g., small translation/rotation simulation)
        altered_tensor = original_tensor + (torch.randn_like(original_tensor) * 0.01)
        
        with torch.no_grad():
            outputs_orig = model(original_tensor)
            outputs_alt = model(altered_tensor)
            
        # Measure the logical distance (KL Divergence or Simple Mean Squared Error)
        # If the distance is high, the model's internal logic is unstable (Adversarial)
        logic_drift = torch.mean((outputs_orig - outputs_alt) ** 2).item()
        return logic_drift, outputs_orig

    def verify(self, model, input_tensor):
        drift, base_output = self.multi_view_check(model, input_tensor)
        
        if drift > self.threshold:
            return "REJECTED (Inconsistent Logic / Potential Attack)", drift
        else:
            return "ACCEPTED (Valid Input Structure)", drift

# --- Local Execution Execution Test ---
# Simulating a random input image tensor [Batch, Channels, Height, Width]
fake_image = torch.randn(1, 3, 224, 224)

engine = BaseEngine()
gate = ValidationGate(threshold=0.05)

# Test clean input
status, drift_score = gate.verify(engine, fake_image)
print(f"Clean Input Status: {status} | Logic Drift Score: {drift_score:.5f}")
# 3. Create an Adversarial Attack Simulator
def generate_adversarial_noise(input_tensor, noise_magnitude=0.35):
    """
    Simulates an adversarial perturbation designed to disrupt 
    the model's geometric alignment without completely destroying the image.
    """
    # Create high-frequency structured noise targeting the tensor space
    noise = torch.sin(input_tensor * 5.0) * noise_magnitude
    adversarial_tensor = input_tensor + noise
    return torch.clamp(adversarial_tensor, -3.0, 3.0)

# --- Executing Day 2 Test Loop ---

# Generate the corrupted adversarial input
corrupted_image = generate_adversarial_noise(fake_image, noise_magnitude=0.4)

# Put the corrupted input through our Validation Gate
corrupted_status, corrupted_drift_score = gate.verify(engine, corrupted_image)

print("--- WEEK 1, DAY 2 METRICS ---")
print(f"Clean Input Status: {status} | Score: {drift_score:.5f}")
print(f"Corrupted Input Status: {corrupted_status} | Score: {corrupted_drift_score:.5f}")
# 4. Implement a Real Gradient-Based Adversarial Attack (FGSM)
def generate_fgsm_attack(model, input_tensor, epsilon=0.15):
    """
    Generates a true adversarial perturbation by calculating the gradient 
    of the model's output loss relative to the input image.
    """
    # Enable gradient tracking on the input tensor
    input_tensor.requires_grad = True
    
    # Run a forward pass
    output = model(input_tensor)
    
    # Target an arbitrary class index (e.g., class 100) to maximize loss against
    target_class = torch.tensor([100]).to(input_tensor.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target_class)
    
    # Zero all existing gradients, then backpropagate to find the input gradients
    model.zero_grad()
    loss.backward()
    
    # Collect the sign of the input gradients
    data_grad = input_tensor.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create the adversarial image by perturbing the original image along the gradient sign
    perturbed_image = input_tensor + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, -3.0, 3.0)
    
    # Detach the tensor so it doesn't carry historical gradients forward
    return perturbed_image.detach()

# --- Executing Day 3 True Attack Loop ---

# Generate a true gradient-based attack
true_adversarial_image = generate_fgsm_attack(engine, fake_image, epsilon=0.2)

# Verify using our current Validation Gate
attack_status, attack_drift_score = gate.verify(engine, true_adversarial_image)

print("--- WEEK 1, DAY 3 METRICS ---")
print(f"True Adversarial Input Status: {attack_status} | Score: {attack_drift_score:.5f}")
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as F
import numpy as np

# 1. Base Engine with a standard pre-trained model
class BaseEngine(nn.Module):
    def __init__(self):
        super(BaseEngine, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()
        
    def forward(self, x):
        return self.model(x)

# 2. Complete Week 1 Advanced Validation Gate
class AdvancedValidationGate:
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        
    def estimate_uncertainty(self, model, x, num_passes=3):
        """
        Runs multiple forward passes with tiny spatial variations 
        to calculate predictive variance (Uncertainty Estimation).
        """
        outputs = []
        with torch.no_grad():
            # Pass 1: Original View
            outputs.append(model(x))
            # Pass 2: Slightly Rotated View (Simulating multi-view physical agreement)
            rotated_x = F.rotate(x, angle=3.0) 
            outputs.append(model(rotated_x))
            # Pass 3: Slightly Shifted View
            shifted_x = F.affine(x, angle=0, translate=[2, 2], scale=1.0, shear=0)
            outputs.append(model(shifted_x))
            
        # Stack outputs along a new dimension: [num_passes, batch, classes]
        stacked_outputs = torch.stack(outputs)
        
        # Calculate the variance across the different spatial views
        variance = torch.var(stacked_outputs, dim=0)
        mean_variance = torch.mean(variance).item()
        return mean_variance, outputs[0]

    def verify(self, model, input_tensor):
        # Scale up the raw logic drift check using spatial transformations
        drift_score, base_output = self.estimate_uncertainty(model, input_tensor)
        
        # Multiply drift by a scaling factor to make variations readable
        scaled_score = drift_score * 100.0
        
        if scaled_score > self.threshold:
            return "REJECTED (Adversarial/Inconsistent Structure)", scaled_score
        else:
            return "ACCEPTED (Valid Physical Structure)", scaled_score

# 3. FGSM Attack Generator
def generate_fgsm_attack(model, input_tensor, epsilon=0.1):
    input_tensor.requires_grad = True
    output = model(input_tensor)
    target_class = torch.tensor([100]).to(input_tensor.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target_class)
    
    model.zero_grad()
    loss.backward()
    
    sign_data_grad = input_tensor.grad.data.sign()
    perturbed_image = input_tensor + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, -3.0, 3.0).detach()

# --- RUNNING THE ENTIRE WEEK 1 VALIDATION ENGINE ---

# Initialize elements
engine = BaseEngine()
gate = AdvancedValidationGate(threshold=1.0) # Threshold set to 1.0 based on scaled variance

# Generate a clean fake image
clean_image = torch.randn(1, 3, 224, 224)

# Test 1: Clean Input Pass
clean_status, clean_score = gate.verify(engine, clean_image)

# Test 2: Generate a true gradient-based attack on that clean image
adversarial_image = generate_fgsm_attack(engine, clean_image.clone(), epsilon=0.15)
attack_status, attack_score = gate.verify(engine, adversarial_image)

print("==== WEEK 1 COMPLETE: VALIDATION METRICS ====")
print(f"Clean Input:      {clean_status} | Scaled Score: {clean_score:.5f}")
print(f"Attack Input:     {attack_status} | Scaled Score: {attack_score:.5f}")
