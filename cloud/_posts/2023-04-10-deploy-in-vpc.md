---
title: "AWS VPC 구성하기(2) - VPC에 인스턴스 배포하기"
published: true
future: true
date: 2023-04-10T17:20:50-04:00
tags:
    - AWS
    - VPC
    - EC2
---

이번 글에서는 이전 글에서 만든 private 네트워크에 인스턴스를 배포해볼 것입니다. 

아키텍처 그림을 보시면 public subnet에는 Bastion Host인 EC2 인스턴스가 있고, private subnet에는 ML 모델이 배포된 EC2 인스턴스가 있습니다.

