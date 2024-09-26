library(ggplot2)
library(tidyverse)
library(car)

##### Load in the data #####

df<-read.csv("replication/replication_complete.csv")
# Make the family column pretty 
df$Family = str_to_title(str_replace(df$Family, "_", " "))


# We're going to be making a lot of log-log plots, so lets make our lives easier:
df$log_test_acc_drop = -1*log((-1*df$test_acc_drop)+1)
df$log_train_lemma_diff = -1*log(-1*df$train_lemma_diff_raw)


##### Plotting all the strong relationships, on the log-log scale #####

# Training size vs. test accuracy drop, log-log scale
df %>% 
  ggplot(aes(log(Goldman_train_size), log_test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Test accuracy drop, log scale") + 
  ggtitle("Test accuracy drop vs. training size") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Training lemmas vs. test accuracy drop, log-log-scale 
df %>% 
  ggplot(aes(log(Goldman_train_lemmas), log_test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. number of training lemmas, log scale") + 
  ylab("Test accuracy drop, log scale") + 
  ggtitle("Test accuracy drop vs. number of training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Training lemma difference vs. test accuracy drop, log-log scale
df %>% 
  ggplot(aes(-log(-1*train_lemma_diff_raw), log_test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Training lemma difference between SIGMORPHON & Goldman et al., log scale") + 
  ylab("Test accuracy drop, log scale") + 
  ggtitle("Test accuracy drop vs. training lemma drop") +
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

# Training size vs. training lemmas, log-log plot
df %>% 
  ggplot(aes(log(Goldman_train_size), log(Goldman_train_lemmas))) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Goldman et al. number of training lemmas, log scale") + 
  ggtitle("Training size vs. training lemmas, log scale") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)



##### Basic stats before the fancy ones #####

# Calculate the basic stats we'll use
correlations <- function(a, b){
  print(cor.test(a, b, method="pearson"))
  print(cor.test(a, b, method="spearman"))
  print(cor.test(a, b, method="kendall"))
}

# Training size
correlations(log(df$Goldman_train_size), df$log_test_acc_drop)

# Training lemmas
correlations(log(df$Goldman_train_lemmas), df$log_test_acc_drop)

# Training lemma drop
correlations(df$log_train_lemma_diff, df$log_test_acc_drop)


##### Co-linearity #####

# Training size vs. training lemmas, log-log
df %>%
  ggplot(aes(log(Goldman_train_size), log(Goldman_train_lemmas))) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Training lemmas, log scale") + 
  ggtitle("Training size vs. training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

correlations(log(df$Goldman_train_size), log(df$Goldman_train_lemmas))

# Training size vs. difference in training lemmas, log-log
df %>%
  ggplot(aes(log(Goldman_train_size), log_train_lemma_diff)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training size, log scale") + 
  ylab("Training lemma difference, log scale") + 
  ggtitle("Training size vs. training lemma drop") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

correlations(log(df$Goldman_train_size), df$log_train_lemma_diff)

# Training lemmas vs. difference in training lemmas, log-log
df %>%
  ggplot(aes(log(Goldman_train_lemmas), log_train_lemma_diff)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training lemmas, log scale") + 
  ylab("Training lemma difference, log scale") + 
  ggtitle("Training lemmas vs. training lemma drop") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )  + 
  stat_smooth(method="lm", color="black", size=0.5, alpha = 0.5)

correlations(log(df$Goldman_train_lemmas), df$log_train_lemma_diff)



##### Pirate Stats! #####

# Fit a linear model to just train size and look at the R^2 & AIC
train_size_lm = lm(log_test_acc_drop ~ log(Goldman_train_size), data = df)
summary(train_size_lm)
AIC(train_size_lm)

# Fit a linear model to just train lemmas and look at the R^2 and AIC
train_lemma_lm = lm(log_test_acc_drop ~ log(Goldman_train_lemmas), data = df)
summary(train_lemma_lm)
AIC(train_lemma_lm)

# Fit a linear model to just lemma drop and look at the R^2 and AIC
lemma_drop_lm = lm(log_test_acc_drop ~ log_train_lemma_diff, data = df)
summary(lemma_drop_lm)
AIC(lemma_drop_lm)

# Fit a linear model to lemmas + train size
lemmas_train_lm = lm(log_test_acc_drop ~ log(Goldman_train_size) + log(Goldman_train_lemmas), data = df)
summary(lemmas_train_lm)
AIC(lemmas_train_lm)
anova(train_lemma_lm, lemmas_train_lm)
vif(lemmas_train_lm)

# Fit a linear model to lemmas + lemma drop
lemmas_drop_lm = lm(log_test_acc_drop ~ log(Goldman_train_lemmas) + log_train_lemma_diff, data = df)
summary(lemmas_drop_lm)
AIC(lemmas_drop_lm)
anova(train_lemma_lm, lemmas_drop_lm)
vif(lemmas_drop_lm)



###### Basic Scatter Plots in case we still want them later ######

# Training size
df %>% 
  ggplot(aes(Goldman_train_size, test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training size") + 
  ylab("Test accuracy drop") + 
  ggtitle("Replication of Goldman et al.") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
        )

# Training lemmas
df %>% 
  ggplot(aes(Goldman_train_lemmas, test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Goldman et al. training lemmas") + 
  ylab("Test accuracy drop") + 
  ggtitle("Test accuracy drop vs. training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )

# Raw difference
df %>% 
  ggplot(aes(train_lemma_diff_raw, test_acc_drop)) + 
  geom_point(aes(colour = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  theme_bw() + 
  xlab("Training lemma difference between SIGMORPHON & Goldman et al.") + 
  ylab("Test accuracy drop") + 
  ggtitle("Test accuracy drop vs. training lemma drop") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  )



##### Effects on test accuracy overall rather than on test #####

Gold <- df %>% 
  select(train=Goldman_train_size, 
         lemmas = Goldman_train_lemmas,
         test = Goldman_test_acc, 
         Family = Family
         ) 
Gold$Type = "Goldman"

Sigm <- df %>% 
  select(train = SIGMORPHON_train_size,
         lemmas = SIGMORPHON_train_lemmas,
         test = SIGMORPHON_test_acc,
         Family = Family
         )
Sigm$Type = "SIGMORPHON"

new_df = rbind(Gold, Sigm)

new_df %>% 
  ggplot(aes(log(train), log(test + 1), color = Type)) +
  geom_point(aes(shape = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  stat_smooth(aes(color=Type), method="lm", size=0.5, alpha = 0.5)+ 
  theme_bw() + 
  xlab("Training size") + 
  ylab("Test accuracy") + 
  ggtitle("Test accuracy vs. training size") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  ) 
correlations(log(df$Goldman_train_size), log(df$Goldman_test_acc + 1))
correlations(log(df$SIGMORPHON_train_size), log(df$SIGMORPHON_test_acc + 1))



new_df %>% 
  ggplot(aes(log(lemmas), log(test + 1), color = Type)) +
  geom_point(aes(shape = Family), size = 5, alpha = 0.5) + 
  scale_color_manual(values=c("turquoise", "purple", "gold")) + 
  stat_smooth(aes(color=Type), method="lm", size=0.5, alpha = 0.5) + 
  theme_bw() + 
  xlab("Training lemmas") + 
  ylab("Test accuracy") + 
  ggtitle("Test accuracy vs. training lemmas") + 
  theme(plot.title = element_text(hjust=0.5, size=18), 
        axis.title.y = element_text(size=14),
        axis.title.x = element_text(size=14),
  ) 

correlations(log(df$Goldman_train_lemmas), log(df$Goldman_test_acc + 1))
correlations(log(df$SIGMORPHON_train_lemmas), log(df$SIGMORPHON_test_acc + 1))

