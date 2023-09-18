- 모델을 학습하다보면 Overfitting(과적합)이 발생할 수 있다.
- 이 경우 가장 단순하게 해결하는 방법은 학습 데이터의 수를 늘리는 것이다.
- 하지만 문제에 따라서 학습 데이터를 구하기가 매우 어려울 수 있다.
- 예를 들어 치매환자의 뇌 MRI 영상 같은 경우 영상 하나를 만들기 위해 건당 수십만원의 비용이 소모된다.
- Overfitting 문제를 해결하기 위해서 여러가지 방법이 쓰일 수 있는데, 그 중 한가지가 Weight decay이다. 
![](http://drive.google.com/uc?export=view&id=1zWmTpoF6YAkeqi6Ac-DiQaNH1iUMOqA3)
- Loss function이 작아지는 방향으로만 단순하게 학습을 진행하면 특정 가중치 값들이 커지면서 위 첫번째 그림처럼 오히려 결과가 나빠질 수 있다.
- Weight decay는 학습된 모델의 복잡도를 줄이기 위해서 학습 중 weight가 너무 큰 값을 가지지 않도록 Loss function에 Weight가 커질경우에 대한 패널티 항목을 집어넣는다.
- 이 패널티 항목으로 많이 쓰이는 것이 L1 Regularization과 L2 Regularization이다.
- Weight decay 를 적용할 경우 위 두번째 그림처럼 Overfitting에서 벗어날 수 있다.