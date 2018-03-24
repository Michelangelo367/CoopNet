import matplotlib.pyplot as plt
plt.switch_backend('agg')

with open('egret/log/cat.log')as f:
	loss = f.read()

loss = [float(num) for num in loss.split()]
epoch = [10*i for i in range(len(loss))]
epoch[0] = 1

plt.title('Descriptor Network Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(epoch, loss)
plt.savefig('egret_loss.png')