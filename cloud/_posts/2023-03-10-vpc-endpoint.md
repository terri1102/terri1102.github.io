---


---

# 목적

>  
> 

# 과정


## 1. 서브넷과 보안 그룹 생성
### 1-1. 프라이빗 서브넷 생성

### 1-2. 보안그룹 생성하기
- private
- public

## 2. VPC Endpoint
### 2-1. VPC Endpoint란

### 2-2. VPC Endpoint 생성


## 3. 네트워크 테스트
### 3-1. 테스트를 위한 EC2 인스턴스 생성
Bastion host
### 3-2. 로드 밸런서 설정
네트워크 로드 밸런서

### 3-3. ECS 생성
Fargate의 경우


# Reference

- [왕초보 탈출 AWS 매뉴얼(2)] Private Subnet 생성: [https://minjii-ya.tistory.com/33?category=946161](https://minjii-ya.tistory.com/33?category=946161)
- Amazon ECR interface VPC endpoints(AWS PrivateLink): [https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html](https://docs.aws.amazon.com/AmazonECR/latest/userguide/vpc-endpoints.html)
- Access an AWS service using an interface VPC endpoint: [https://docs.aws.amazon.com/vpc/latest/privatelink/create-interface-endpoint.html#create-interface-endpoint](https://docs.aws.amazon.com/vpc/latest/privatelink/create-interface-endpoint.html#create-interface-endpoint)
- 프라이빗 서브넷의 Fargate에서 Amazon ECS 태스크를 실행하려면 어떻게 해야 하나요?: ****[https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-fargate-tasks-private-subnet/](https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-fargate-tasks-private-subnet/)
- Amazon ECS에서 “unable to pull secrets or registry auth” 오류를 해결하려면 어떻게 해야 합니까?: [https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-unable-to-pull-secrets/](https://aws.amazon.com/ko/premiumsupport/knowledge-center/ecs-unable-to-pull-secrets/)
https://dingrr.com/blog/post/%EC%9D%B4%EC%A0%9C-%EB%A7%89-%EC%8B%9C%EC%9E%91%ED%95%9C-%EC%8A%A4%ED%83%80%ED%8A%B8%EC%97%85%EC%9D%84-%EC%9C%84%ED%95%9C-aws-%EC%95%84%ED%82%A4%ED%85%8D%EC%B2%98-aws%EB%A5%BC-%EB%8D%94-aws-%EB%8B%B5%EA%B2%8C