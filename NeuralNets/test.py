import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import PiattiCNN
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

def format_parameter_count(count):
    """Format parameter count (e.g., 1.2M, 345K)"""
    if count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)

def test_model(model, test_loader, device, class_names):
    """Evaluate model on test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    # Classification report
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {accuracy:.2f}%\n")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - Test Accuracy: {accuracy:.2f}%')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\n✓ Confusion matrix saved to 'confusion_matrix.png'")
    
    return accuracy, all_preds, all_labels

if __name__ == '__main__':
    # Note that test transformation are consistent and reproducible (deterministic)
    # No random, no flip, no jitter (different from training)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}\n")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PiattiCNN(num_classes=len(test_dataset.classes)).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('PiattiVL-1232kresnet.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model: PiattiCNN")
    print(f"Parameters: {format_parameter_count(num_params)} ({num_params:,})")
    print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")
    print(f"Trained for {checkpoint['epoch']+1} epochs\n")
    
    # Test
    test_accuracy, preds, labels = test_model(model, test_loader, device, test_dataset.classes)
    
    print(f"\n{'='*60}")
    print(f"FINAL TEST ACCURACY: {test_accuracy:.2f}%")
    print(f"{'='*60}\n")