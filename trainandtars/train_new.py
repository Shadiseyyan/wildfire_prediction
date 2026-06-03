import torch
import os
# Import your training function from your main architecture file
from maine import train_on_trainandtar 

print("Training new weights on the ERA5 dataset...")
# Point it to your new folder and run it for a few epochs
model, cfg = train_on_trainandtar(
    train_dir=os.path.join("..", "test_and_tar"),
    n_epochs=4,        # Keep it short just to test!
    batch_size=2, 
    lr=3e-4
)

# Save the new, perfectly fitted weights!
torch.save(model.state_dict(), "camp_fire_weights.pth")
print("Saved new weights as 'camp_fire_weights.pth'!")