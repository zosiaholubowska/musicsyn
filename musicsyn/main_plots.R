rm(list=ls()) 
library(scales)
library(readxl)
library(tidyr)
library(plyr)
library(dplyr)
library(ggplot2)
library(Rmisc)
library(tidyverse)
library(hrbrthemes)
library(ggpubr)
library(psych)
library(slider)
library(rstatix)
library(viridis) 

setwd("~/Documents/PhD/3_experiment/experiment")

data <- read_csv("main.csv")

data %>%
  ggplot(aes(x=condition, y=miss, fill=condition))+
  geom_boxplot()+
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(legend.position="none",
    plot.title = element_text(size=11))

A <- data %>%
  ggplot( aes(x=condition, y=d_prime, fill=condition)) +
  geom_violin() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("d-prime") +
  xlab("")

B <- data %>%
  ggplot( aes(x=condition, y=ff1, fill=condition)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("F-score") +
  xlab("")

C <- data %>%
  ggplot( aes(x=stimulus, y=d_prime, fill=condition)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("d-prime / stimulus") +
  xlab("")

D <- data %>%
  ggplot( aes(x=stimulus, y=ff1, fill=condition)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("f-score / stimulus") +
  xlab("")

E <- data %>%
  ggplot( aes(x=subject, y=d_prime, fill=condition)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("d-prime / subject") +
  xlab("")

F <- data %>%
  ggplot( aes(x=subject, y=ff1, fill=condition)) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("f-score / subject") +
  xlab("") +
  scale_fill_manual(values=c("skyblue", "cornflowerblue"))


G <- data %>%
  ggplot( aes(x=participant, y=ff1, fill="skyblue")) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("f-score / subject") +
  xlab("")+
  scale_fill_manual(values=c("skyblue", "cornflowerblue"))

H <- data %>%
  ggplot( aes(x=participant, y=d_prime, fill="skyblue")) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("d_prime / subject") +
  xlab("") +
  scale_fill_manual(values=c("skyblue", "cornflowerblue"))


data %>%
  ggplot( aes(x=condition, y=hit_rate, fill="skyblue")) +
  geom_boxplot() +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme_ipsum() +
  theme(
    plot.title = element_text(size=11)
  ) +
  ggtitle("d_prime / subject") +
  xlab("") +
  scale_fill_manual(values=c("skyblue", "cornflowerblue"))




ggarrange(A, B, C, D, E, F,
          ncol = 2, nrow = 3)


data <- read_csv("musicsyn/results.csv")

data <- data[, -1]

data <- data %>%
  filter(visual_cue == 1)
data <- data %>%
  filter(answer == 1)
data$condition <- ifelse(data$boundary == 0, "no_boundary", "boundary")


p <- ggboxplot(data, x = "condition", y = "time_difference",
               fill = "condition", palette = "jco",
               add = "jitter")
#  Add p-value
p + stat_compare_means()
# Change method
p + stat_compare_means(method = "t.test")


p <- ggboxplot(data, x = "condition", y = "hit_rate",
               fill = "condition", palette = "jco",
               add = "jitter")
#  Add p-value
p + stat_compare_means()
# Change method
p + stat_compare_means(method = "t.test")
