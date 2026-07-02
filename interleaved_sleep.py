import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define a simple, flexible Cognitive Network
class SimpleBrain(nn.Module):
    def __init__(self):
        super(SimpleBrain, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.fc(x)

# 2. Generate two completely distinct logical tasks (Task A and Task B)
# Task A: Predict if numbers are mostly positive
X_A = torch.randn(100, 10)
Y_A = (torch.sum(X_A, dim=1) > 0).long()

# Task B: Predict if the numbers have an even sum of absolute values
X_B = torch.randn(100, 10)
Y_B = (torch.sum(torch.abs(X_B), dim=1) > 10.0).long()

# 3. Define the Interleaved Sleep Algorithm
def simulate_sleep_cycle(main_brain, task_a_data, task_b_data, epochs=20):
    """
    Simulates interleaved replay during a sleep cycle. 
    Replays historical memories (Task A) alongside new memories (Task B)
    to gently weave them into a single, stable weight matrix.
    """
    print("\n[System Alert] Initiating Daily Maintenance Phase (Sleep Cycle)...")
    optimizer = optim.Adam(main_brain.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    X_A, Y_A = task_a_data
    X_B, Y_B = task_b_data
    
    for epoch in range(epochs):
        # Interleave the batches (mixing past and present experiences)
        optimizer.zero_grad()
        
        out_A = main_brain(X_A)
        out_B = main_brain(X_B)
        
        loss = criterion(out_A, Y_A) + criterion(out_B, Y_B)
        loss.backward()
        optimizer.step()
        
    print("[System Alert] Synaptic pruning and consolidation complete. Brain is awake.")

# --- RUNNING THE SYSTEM TIMELINE ---

brain = SimpleBrain()
criterion = nn.CrossEntropyLoss()

# STEP 1: Train exclusively on Task A (Day 1)
optimizer = optim.Adam(brain.parameters(), lr=0.05)
for _ in range(30):
    optimizer.zero_grad()
    loss = criterion(brain(X_A), Y_A)
    loss.backward()
    optimizer.step()

# Check initial competence on Task A
with torch.no_grad():
    acc_A_init = (torch.argmax(brain(X_A), dim=1) == Y_A).float().mean().item() * 100
print(f"Competence on Task A after initial training: {acc_A_init:.1f}%")

# STEP 2: Force-train on Task B *WITHOUT* Sleep (Simulating standard AI Catastrophic Forgetting)
for _ in range(30):
    optimizer.zero_grad()
    loss = criterion(brain(X_B), Y_B)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    acc_A_forgot = (torch.argmax(brain(X_A), dim=1) == Y_A).float().mean().item() * 100
    acc_B_forgot = (torch.argmax(brain(X_B), dim=1) == Y_B).float().mean().item() * 100
print("\n--- Force-Learning New Task Without Sleep ---")
print(f"Task A Competence (CRASHED): {acc_A_forgot:.1f}%")
print(f"Task B Competence (LEARNED): {acc_B_forgot:.1f}%")

# STEP 3: Run the Custom Interleaved Sleep Cycle to restore balance
simulate_sleep_cycle(brain, (X_A, Y_A), (X_B, Y_B), epochs=40)

with torch.no_grad():
    acc_A_restored = (torch.argmax(brain(X_A), dim=1) == Y_A).float().mean().item() * 100
    acc_B_restored = (torch.argmax(brain(X_B), dim=1) == Y_B).float().mean().item() * 100
print("\n==== WEEK 2 COMPLETE: MEMORY CONSOLIDATION METRICS ====")
print(f"Restored Task A Competence: {acc_A_restored:.1f}%")
print(f"Restored Task B Competence: {acc_B_restored:.1f}%")
