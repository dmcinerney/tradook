from Loading import *
from torch import optim
from jack_net import *

image_size = 50

def get_test_accuracy(net):
    transform = transforms.Compose([transforms.Scale((net.image_size, net.image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='test.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    accuracy = 0
    iter_num = 1
    for images, labels in iter(dataloader):
        if torch.cuda.is_available():
            inputs = Variable(images, volatile=True).cuda()
        else:
            inputs = Variable(images, volatile=True)

        predicted = net.predict(inputs).data.cpu()
        equality = (labels == predicted.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        iter_num += 1
    print 'test accuracy: ', float(accuracy) / float(iter_num)





def train():
    global image_size
    if torch.cuda.is_available():
        print 'running with cuda'
        net = ConvNet(image_size).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        net = ConvNet(image_size)
        criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.Scale((net.image_size, net.image_size)), transforms.ToTensor()])
    dataset = image_loading(csv_file='train.txt', root_dir='LetterImages/',  transformation = transform)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=12)

    learning_rate = 1e-4
    optimizer = optim.Adam(net.parameters(), learning_rate)

    train_loss_list = list()
    testing_accuracy_list = list()
    steps = 1
    running_loss = 0
    print_every = 500
    for epoch in range(5):
        print epoch
        for images, labels in iter(dataloader):

            labels = np.array(map(int, labels))
            # print type(labels)
            labels = torch.from_numpy(labels)
            # .view(labels.shape[0], -1)
            labels = labels.type(torch.FloatTensor)
            # print labels.size()
            # image = images.resize(images.size()[0], net.image_size**2)
            # images = images.view(images.numel())
            if torch.cuda.is_available():
                inputs = Variable(images).cuda()
                target = Variable(labels).cuda()
            else:
                inputs = Variable(images)
                target = Variable(labels)
            target = target.type(torch.LongTensor)
            optimizer.zero_grad()
            # print inputs.size()
            output = net.forward(inputs)
            # print output.size()
            # print target.size()
            # print output.size()
            # print output
            # print type(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            train_loss_list.append(loss.data[0])

            if steps % print_every == 0:
                print 'training loss ', float(running_loss) / float(steps)
                testing_accuracy_list.append(get_test_accuracy(net))
            steps+=1

    x_training = np.linspace(0, iter_num, len(training_loss_list))
    plt.plot(x_training, training_loss_list)
    plt.title('training loss')
    plt.savefig('jack_training_loss.png')
    plt.clf()

    x_testing = np.linspace(0, iter_num, len(testing_accuracy_list))
    plt.plot(x_testing, testing_accuracy_list)
    plt.title('testing accuracy')
    plt.savefig('jack_testing_accuracy.png')
    plt.clf()


train()
