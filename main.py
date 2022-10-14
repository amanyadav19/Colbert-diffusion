from dataset import MetaQADataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
from message_passing import SAGEConv
import torch
from GNN import Net
import random

metaQAdata = MetaQADataset('/mnt/infonas/data/amanyadav/MetaQA')
metaQAdata_train = metaQAdata.shuffle()
# print(type(metaQAdata_train))
# print(len(metaQAdata_train.x))

# batch_size= 1024
train_loader = DataLoader(metaQAdata_train)

device = torch.device('cpu')
model = Net(metaQAdata_train.x.size(dim=0), metaQAdata_train.edge_index[0].size(dim=0)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
crit = torch.nn.BCELoss()

def negative_sample(kg, answer_node_entity):
    new_kg = []
    all_answer_nodes = []
    for elem in answer_node_entity:
        for node in elem[0]:
            all_answer_nodes.append(elem[0])
    all_answer_nodes = all_answer_nodes.sort()
    for node in kg.x:
        found = False
        for elem in all_answer_nodes:
            if(node == elem):
                found = True
        if not found:
            new_kg.append(node)
    negative_sample = random.choice(new_kg)
    return ([negative_sample], [], [])

def train():
    model.train()

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, e = model(data)
        label = data.y.to(device)
        loss = crit(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


def evaluate(loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    
    return roc_auc_score(labels, predictions)

for epoch in range(1):
    loss = train()
    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)    
    test_acc = evaluate(test_loader)
    print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
          format(epoch, loss, train_acc, val_acc, test_acc))