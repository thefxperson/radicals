import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, 5, stride=1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    #computes the forward pass of the network
    def forward(self, x):
        #x_dim = 1x28x28
        x = F.relu(self.conv1(x))
        #x_dim = 20x24x24
        x = F.max_pool2d(x, 2, 2)
        #x_dim = 20x12x12
        x = F.relu(self.conv2(x))
        #x_dim = 50x8x8
        x = F.max_pool2d(x, 2, 2)
        #x_dim = 50x4x4
        x = x.view(-1, 4*4*50)
        #x_dim = 800
        x = F.relu(self.fc1(x))
        #x_dim = 500
        x = self.fc2(x)
        #x_dim = 10
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          #send data and labels to proper device (CPU or GPU)
          data, target = data.to(device), target.to(device)
          #reset gradients
          optimizer.zero_grad()
          #run inference
          output = model(data)
          #compute negative log likelihood loss
          loss = F.nll_loss(output, target)
          #compute gradients
          loss.backward()
          #run backpropigation
          optimizer.step()
          #update human on machine status
          if batch_idx % args.log_interval == 0:
              print("Epoch: {}\tBatch: {} ({:.0f}%)\tLoss: {:.6f}".format(
                  epoch + 1, batch_idx, 100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    n_correct = 0
    #do not compute gradients because we are not training
    with torch.no_grad():
        for data, target in test_loader:
            #send data and labels to proper device (CPU or GPU)
            data, target = data.to(device), target.to(device)
            #run inference
            output = model(data)
            #compute negative log likelihood loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            #compute the index of the classified character from the probability outputs
            pred = output.argmax(dim=1, keepdim=True)
            #find the number of images the model properly classified
            n_correct += pred.eq(target.view_as(pred)).sum().item()

    #find the mean of the loss
    test_loss /= len(test_loader.dataset)

    print("Test Set: \tMean Loss: {:.4f}, Accuracy: {:.4f}".format(
        test_loss, n_correct / len(test_loader.dataset)))

def main():
    #training settings
    parser = argparse.ArgumentParser(description="MNIST Configuration")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N")
    parser.add_argument("--epochs", type=int, default=10, metavar="N")
    parser.add_argument("--learning-rate", type=float, default=0.01, metavar="LR")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-model", action="store_true", default=False)
    args = parser.parse_args()

    #set up device -- GPU or CPU?
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    #seed torch for random numbers
    torch.manual_seed(1)

    #load datasets
    train_loader = torch.utils.data.DataLoader(datasets.MNIST("../data", train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                          ])),
                            batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST("../data", train=False,
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize((0.1307,), (0.3081,))
                                                          ])),
                            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    #send model to proper device (GPU or CPU)
    model = Net().to(device)
    #create optimizer
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    #train model
    for epoch in range(args.epochs):
        if epoch % 5 == 0 and epoch != 0:
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate/2, momentum=0.9)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    #save model
    print("Saving Model")
    torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":
    main()
