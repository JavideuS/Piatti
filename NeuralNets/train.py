import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from model import PiattiCNN
from tqdm import tqdm  # Prettier visuals for loops

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
    epoch_acc = 100. * train_correct / train_total
    return train_loss / len(train_loader), epoch_acc


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    epoch_acc = 100. * val_correct / val_total
    return val_loss / len(val_loader), epoch_acc


if __name__ == '__main__':
    # Transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # More aggressive
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),              # ← New
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1), # Stronger + hue
        transforms.RandomGrayscale(p=0.1),          # ← New
        transforms.RandomPerspective(p=0.2),        # ← New
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),            # ← New: random patches
    ])
    
    # Load dataset
    dataset = datasets.ImageFolder(root='data/train', transform=transform_train)
    print(f'Classes: {dataset.classes}')
    
    # Split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=0, stratify=dataset.targets
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                   num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                 num_workers=4, pin_memory=True)
    
    print(f'Training: {len(train_dataset)}, Validation: {len(val_dataset)}')
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PiattiCNN(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0  # Predefined baseline
    
    for epoch in range(num_epochs):
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_params': num_params,
                'num_classes': len(dataset.classes),
                'class_names': dataset.classes,
            },f'PiattiVL-{num_params//1000}kresnet.pth')
            print(f'✓ Saved best model: {val_acc:.2f}%')
    
    print(f'\nBest validation accuracy: {best_val_acc:.2f}%')