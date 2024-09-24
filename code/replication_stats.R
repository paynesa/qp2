library(ggplot2)
library(tidyverse)

##### Load in the data #####

df<-read.csv("replication/replication_complete.csv")
# Make the family column pretty 
df$Family = str_to_title(str_replace(df$Family, "_", " "))
# Smooth + log scale the test_acc_drop column
df$log_test_acc_drop = -1*log((-1*df$test_acc_drop)+1)


###### The Role of Training Size ######

# Here's the basic scatter plot, Goldman style
df %>% 
  ggplot(aes(Goldman_train_size, test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. training size") + 
  ylab("Test accuracy drop") + 
  ggtitle("Replication of Goldman et al.") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
        )

# Now for the log-log plot
df %>% 
  ggplot(aes(log(Goldman_train_size), log_test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Test accuracy drop, log scale") + 
  ggtitle("Replication of Goldman et al.") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Let's fit a linear model
train_size_lm = lm(log_test_acc_drop ~ log(Goldman_train_size), data = df)
summary(train_size_lm)

# Now let's do some stats
cor.test(df$log_test_acc_drop, log(df$Goldman_train_size), method="pearson")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_size), method="spearman")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_size), method="kendall")


##### The Role of Training Lemmas #####

# Here's the basic scatter plot, Goldman style
df %>% 
  ggplot(aes(Goldman_train_lemmas, test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. number of training lemmas") + 
  ylab("Test accuracy drop") + 
  ggtitle("Test accuracy drop vs. number of training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )

# Now for the log-log plot
df %>% 
  ggplot(aes(log(Goldman_train_lemmas), log_test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. number of training lemmas, log scale") + 
  ylab("Test accuracy drop, log scale") + 
  ggtitle("Test accuracy drop vs. number of training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Let's fit a linear model
train_lemmas_lm = lm(log_test_acc_drop ~ log(Goldman_train_lemmas), data = df)
summary(train_lemmas_lm)

# Now let's do some stats
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="pearson")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="spearman")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="kendall")

##### The Relationship Between Training Lemmas and Training Size #####

# Basic scatterplot to start
df %>% 
  ggplot(aes(Goldman_train_size, Goldman_train_lemmas)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. training size") + 
  ylab("Goldman et al. number of training lemmas") + 
  ggtitle("Training size vs. training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Log-log plot
df %>% 
  ggplot(aes(log(Goldman_train_size), log(Goldman_train_lemmas))) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Goldman et al. number of training lemmas, log scale") + 
  ggtitle("Training size vs. training lemmas, log scale") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Let's fit the linear model again
train_vs_lemmas = lm(log(Goldman_train_lemmas) ~ log(Goldman_train_size), data = df)
summary(train_vs_lemmas)

# Now let's do some stats
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="pearson")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="spearman")
cor.test(df$log_test_acc_drop, log(df$Goldman_train_lemmas), method="kendall")