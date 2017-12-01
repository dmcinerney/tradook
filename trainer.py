from Loading import *
from torch import optim

image_size = 50

def train():
    net = Net(image_size)

    global image_size
    transform = transforms.Compose([transforms.Scale((image_size, image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='train.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    steps = 0
    for epoch in range(10):
        print epoch
        for images, labels in iter(dataloader):
            steps+=1

            inputs = Variable(images)
            target = Variable(labels)
            optimizer.zero_grad()

            output = net.forward(inputs)
            loss = criterion(output, targets)
