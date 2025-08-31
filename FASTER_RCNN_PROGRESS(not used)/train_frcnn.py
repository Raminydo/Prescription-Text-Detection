

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from datase_prepration import TextDataset




IMAGE_DIR = 'data/test'
ANNOTATION_FILE = 'FASTER_RCNN_PROGRESS/annotations.json'
MODEL_SAVE_PATH = 'fasterrcnn_model.pt'
CONFIDENCE_THRESHOLD = 0.7
NUM_CLASSES = 2
EPOCHS = 10
BATCH_SIZE = 2



dataset = TextDataset(IMAGE_DIR, ANNOTATION_FILE)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
model = fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

print("Starting training...")
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0
    for images, targets in data_loader:
        try:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
        except:
            continue
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")



torch.save(model, MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")