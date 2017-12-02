from Loading import *
from torch import optim
from jack_net import *

image_size = 50

def get_test_accuracy(net):
    transform = transforms.Compose([transforms.Scale((net.image_size, net.image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='test.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    for images, labels in iter(dataloader):
        inputs = Variable(images)
        target = Variable(labels)

        output = net.forward(inputs)



def train():
    global image_size
    net = ConvNet(image_size)

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

            print type(images)
            print labels
            labels = np.array(map(int, labels))
            # print type(labels)
            labels = torch.from_numpy(labels).view(labels.shape[0], -1)
            labels = labels.type(torch.FloatTensor)
            print labels.size()
            # image = images.resize(images.size()[0], net.image_size**2)
            images = images.view(images.numel())
            inputs = Variable(images)
            target = Variable(labels)
            optimizer.zero_grad()
            print inputs.size()
            output = net.forward(inputs)
            print type(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            # if steps % print_every == 0:
            #     print 'test accuracy for step #', steps, '= ', get_test_accuracy(net)
            steps+=1
            break

train()
