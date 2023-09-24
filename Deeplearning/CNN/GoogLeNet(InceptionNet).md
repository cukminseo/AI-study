## Reference
- ILSVRC 2014에서  VGGNet을 제치고 1등
- 이름의 유래는 거의 대부분 구글직원이라+ 초기 LeNet의 이름을 따라서 Goog+LeNet
- 1 by 1 conv layer의 사용이나 depth를 늘려 성능개선을 꽤하는 듯 VGGNet과 유사한점이 있다.
- VGGNet과 다른점은 inception 모듈이라는 독특한 구조를 사용



> [[GoogLeNet]Going deeper with convolutions (2014)](https://arxiv.org/pdf/1409.4842.pdf)

  
#### Abstract
- 이 모델의 중점적 특징은 연산시 컴퓨팅 리소스의 활용이 개선되었다.
- 성능 최적화를 위해, Hebbian principle과 multi-scale processing의 직관을 사용하였다.
- 이 구조를 GoogLeNet이라 부르며, 22개의 layer를 가진다.

# Introduction

- 지난 3년간 CNN분야의 많은 발전이 있었는데, 그것은 단지 하드웨어 성능의 증가, 커진 dataset, 큰 모델때문이 아니라 새로운 아이디어와 알고리즘, 개선된 신경망 구조때문이였다.
- GoogLeNet은 2년전의 AlexNet보다 파라미터가 12배 적었음에도 불구하고, 더욱 상당히 정확했다. 
- power나 메모리사용량에 한계가 있는 mobile이나 임베디드 컴퓨팅환경에서는 효율적인 알고리즘의 중요성이 대두되기때문에, 그러한 곳에서도 현실적으로 적절히 사용하능하게끔 추론시간에 1.5 billion 이하의 연산만을 수행하도록 설계하였다.
- GoogLeNet의 코드네임 Inception은 Network in Network라는 논문과 유명한 인터넷 밈 "we need to go deeper"에서 유래되어 착안하였다.(인셉션 영화 대사)
- 이때 deep은 두가지 의미를 갖는다.
	1. Inception module의 형태로 새로운 차원의 구조 도입
	2. 네트워크 깊이가 깊어졌다는 직접적인 의미
# Related Work
- LeNet을 필두로, CNN은 표준의 구조를 갖는다.