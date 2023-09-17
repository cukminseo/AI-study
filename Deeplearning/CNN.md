- CNN은 이미지와 같이 지역적인 패턴과 구조가 중요한 데이터를 처리하기 위해 설계된 신경 망 모델이다.
- 이미지를 flatten 하여 MLP학습은 가능하나, 펼치는 행위에서 이미지의 지역 정보(topological information)가 손실 되며, 추상화 없이 바로 연산을 하므로 학습 시간과 능률에 있어 매우 비효율적이다.
- 때문에 이미지의 지역 정보를 잘 살릴 수 있는 receptive field concept를 이용한 CNN이 도입되었다

- 네트워크가 깊어지면(layer가 많아지면) 성능 향상의 가능성이 생긴다.

  - 중첩된 convolution 연산으로 인해 상위 레이어 각 뉴런 입력의 receptive field는 점점 커지게 되고, 따라서 중요한 맥락정보(Contextual Information)를 학습할 수 있고, 복잡한 패턴과 같은 고수준 특징을 얻을 수 있기 때문이다.

- 가능성이 생긴다는 것은, 일반적으로 layer를 많이 쌓아나가는 것은 학습 데이터를 과하게 학습하는 overfitting의 원인이 되기 때문이다.

- 그렇기 때문에 layer를 많이 쌓으면서 과적합을 방지할 수 있는 방법을 제안하여 CNN성능 향상 가능성을 실현해 온 게 CNN의 발전 방향성 중 하나 이다.

  
 - [[AlexNet]]
	 - [AlexNet paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) -2012
	 - [AlexNet paper review](https://velog.io/@kms39273/CNNAlexNet-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)

- [[VGG]]
	- [VGG paper](https://arxiv.org/abs/1409.1556)-2014

- [[InceptionNet]]
	- [InceptionNet paper](https://arxiv.org/abs/1409.4842)-2014

- [[ResNet]]
	- [ResNet paper](https://arxiv.org/abs/1512.03385)-2015
	- 
- [[DenseNet]]
	- [DenseNet paper](https://arxiv.org/abs/1608.06993)-2017

- [[EfficientNet]]
	- [EfficientNet paper](https://arxiv.org/abs/1905.11946)-2019