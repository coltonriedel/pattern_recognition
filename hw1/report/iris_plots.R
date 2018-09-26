#!/usr/bin/env Rscript

library(dplyr)
library(ggplot2)
library(tikzDevice)

iris_train <- read.csv("../../datasets/iris_train.csv")

tikz('iris_train.tex', width=6, height=3)

ggplot(iris_train,
  aes(y=Petallength, x=Petalwidth,
      shape=factor(label), col=as.factor(label))) +
  scale_color_manual(values=c("blue", "red2")) +
  scale_shape_discrete(solid=F) +
  geom_point(size=3.0) +
  ggtitle("Iris Training Dataset") +
  xlab("Petal Width") +
  ylab("Petal Length") +
  ylim(0,7) +
  theme(legend.position="none") +
  guides(fill=FALSE, color=FALSE) +
  stat_function(fun = function(x) 4.85366 - (3.19512 * x), colour="black") +
  annotate("text", label="Perceptron boundary", x=1.75, y=1) +
  theme_bw() +
  theme(legend.position="none")

dev.off()

iris_test <- read.csv("../../datasets/iris_test.csv")

tikz('iris_test.tex', width=6, height=3)

ggplot(iris_test,
  aes(y=Petallength, x=Petalwidth,
      shape=factor(label), col=as.factor(label))) +
  scale_color_manual(values=c("blue", "red2")) +
  scale_shape_discrete(solid=F) +
  geom_point(size=3.0) +
  ggtitle("Iris Test Dataset") +
  xlab("Petal Width") +
  ylab("Petal Length") +
  ylim(0,7) +
  theme(legend.position="none") +
  guides(fill=FALSE, color=FALSE) +
  stat_function(fun = function(x) 4.85366 - (3.19512 * x), colour="black") +
  annotate("text", label="Perceptron boundary", x=1.75, y=1) +
  theme_bw() +
  theme(legend.position="none")

dev.off()
