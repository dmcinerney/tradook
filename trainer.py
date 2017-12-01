from Loading import *
from torch import optim

image_size = 50

def get_test_accuracy(net):
    transform = transforms.Compose([transforms.Scale((net.image_size, net.image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='test.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    for images, labels 


def train():
    global image_size
    net = Net(image_size)

    transform = transforms.Compose([transforms.Scale((net.image_size, net.image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='train.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    steps = 1
    running_loss = 0
    print_every = 500
    for epoch in range(5):
        print epoch
        for images, labels in iter(dataloader):


            inputs = Variable(images)
            target = Variable(labels)
            optimizer.zero_grad()

            output = net.forward(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if steps % print_every == 0:
                print 'test accuracy for step #', steps, '= ', get_test_accuracy(net)
            steps+=1
