---
title: "AWS VPC 구성하기(1) - Private Subnet"
published: true
date: 2023-04-02T00:20:50-04:00
tags:
    - AWS
    - VPC
---

AWS에서 VPC 설정을 하는 방법에 대해서 정리+공부하기 위해서 글을 쓰게 되었습니다. ML 모델을 배포할 때는 주로 VPC 내부에서만 접근 가능한 Private Subnet에 배포를 하게 됩니다. 
이번 편에서는 Private subnet을 구성하고 추후 이어지는 글에서는 Bastion Host를 통해서 Private Subnet에 배포된 ML 서비스에 접근하는 방법을 정리하고자 합니다.


# 과정
1. 서브넷 만들기
2. 라우팅 테이블 만들기
3. 서브넷에 라우팅 테이블 연결하기


## 1. 서브넷 생성하기
AWS의 VPC 콘솔에 들어오면 기본으로 생성된 VPC가 있습니다. 서울 리전(ap-northeast-2)에는 기본적으로 4개의 가용 영역이 있고, 각 가용 영역마다 서브넷이 하나씩 생성되어 있습니다. 

![vpc](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/vpc.png)


![subnets](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/subnets.png)

라우팅 테이블에 들어가면 기본 라우팅 테이블이 있는데, 명시적 연결이 없는 서브넷의 경우 이 라우팅 테이블을 이용하게 됩니다.

![rtb](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/route_table.png)
이제 프라이빗 서브넷을 생성하겠습니다.

![subnet_created](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/subnet_created.png)
일단 default VPC에 새로운 서브넷을 생성해줍니다. 저는 가용 영역을 이름에 넣고 싶어서 private-subnet-2a로 이름을 붙였습니다.

![create_subnet2](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/create_subnet2.png)

<br>

### 2. 라우팅 테이블 만들기
VPC 콘솔에서 왼쪽 네비게이션 바에서 라우팅 테이블을 선택한 다음 라우팅 테이블 생성 버튼을 클릭합니다.
![rtb](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/route_table.png)

라우팅 테이블에 적당한 이름을 붙이고 VPC를 선택해줍니다.
![create_rtb](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/create_routetable.png)

라우팅 테이블이 생성되면 서브넷 연결 탭을 클릭합니다.
![private_rtb](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/private_rtb.png)


<br>

### 3. 라우팅 테이블을 서브넷에 연결하기
이제 private-rtb 라우팅 테이블을 private-subnet-2a에 연결하겠습니다. 
서브넷 연결 탭에서 만들어 둔 private-subnet-2a를 체크한 다음 연결 저장을 클릭해서  라우팅 테이블을 붙입니다.

![connect_to_subnet](https://raw.githubusercontent.com/terri1102/blog_images/main/cloud/connect_to_subnet.png)


현재 private-rtb의 라우팅 테이블은 VPC 내부 통신만 가능한 상태입니다. 나중에 VPC Endpoint를 추가해서 외부 AWS 리소스와 통신할 수 있게 할 것입니다. 

# Reference

- [왕초보 탈출 AWS 매뉴얼(2)] Private Subnet 생성: [https://minjii-ya.tistory.com/33?category=946161](https://minjii-ya.tistory.com/33?category=946161)
- Amazon ECR interface VPC endpoints(AWS PrivateLink): [https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html](https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html)
- Access an AWS service using an interface VPC endpoint: [https://docs.aws.amazon.com/vpc/latest/privatelink/create-interface-endpoint.html#create-interface-endpoint](https://docs.aws.amazon.com/vpc/latest/privatelink/create-interface-endpoint.html#create-interface-endpoint)
- 프라이빗 서브넷의 Fargate에서 Amazon ECS 태스크를 실행하려면 어떻게 해야 하나요?: ****[https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-fargate-tasks-private-subnet/](https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-fargate-tasks-private-subnet/)
- Amazon ECS에서 “unable to pull secrets or registry auth” 오류를 해결하려면 어떻게 해야 합니까?: [https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-unable-to-pull-secrets/](https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-unable-to-pull-secrets/)
https://dingrr.com/blog/post/%EC%9D%B4%EC%A0%9C-%EB%A7%89-%EC%8B%9C%EC%9E%91%ED%95%9C-%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85%EC%9D%84-%EC%9C%84%ED%95%9C-aws-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-aws%EB%A5%BC-%EB%8D%94-aws-%EB%8B%B5%EA%B2%8C