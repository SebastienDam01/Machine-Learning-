# Fonctions de prédiction aussi performantes que possible 
# Pour les données de location de vélos, il faudra également 
# déterminer quelles sont les variables qui influent le plus sur le nombre de locations,
# et analyer le sens de cette influence.

setwd("C:/Users/sebas/Desktop/Supérieur/Branche/GI05/SY19/TP/TP 10")

phonemes <- read.table(file="data/parole_train.txt", sep="")
letter <- read.table(file="data/letters_train.txt", sep="")
bike <- read.csv(file = "data/bike_train.csv")

table(phonemes$y)
table(letter$Y)
plot(bike$instant, bike$cnt)

# Phonèmes

missing_val<-data.frame(apply(phonemes,2,function(x){sum(is.na(x))}))
names(missing_val)[1]='missing_val'
missing_val

# ACP

pca<-princomp(phonemes[, 1:256])
Z<-pca$scores
lambda<-pca$sdev^2
plot(cumsum(lambda)/sum(lambda),type="l",
     xlab="Composantes principales",
     ylab="Proportion de variance expliquée", 
     main="Proportion de variance expliquée selon les composantes principales")

# http://www.sthda.com/french/articles/38-methodes-des-composantes-principales-dans-r-guide-pratique/79-acp-dans-r-prcomp-vs-princomp/#methodes-generales-concernant-lacp

library(factoextra)
res.pca <- prcomp(phonemes[, 1:256], scale = TRUE)

summary(res.pca)

# Valeurs propres
eig.val <- get_eigenvalue(res.pca)
eig.val

# % de variances expliquées par rapport au nombre de PC
fviz_eig(res.pca)

# Graphique des individus. Les individus similaires sont groupés ensemble.
fviz_pca_ind(res.pca,
             col.ind = "cos2", # Colorer par le cos2
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE
)

# Graphique des variables. Coloration en fonction de la contribution des variables.
fviz_pca_var(res.pca,
             col.var = "contrib", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     
)

# The first 50 PC explains around 91.04%. Adding another 50 PC would yield 94.67%.
# So we choose the first xx PC for the next part.

phonemes_pc <- res.pca$x
phonemes_pc <- phonemes_pc[,1:50]
phonemes_pc <- as.data.frame(phonemes_pc)

# Letter recognition

missing_val<-data.frame(apply(letter,2,function(x){sum(is.na(x))}))
names(missing_val)[1]='missing_val'
missing_val

plot(letter[1:1000, 2:9], col=length(unique(letter$Y)))
plot(letter[1:1000, 10:17], col=length(unique(letter$Y)))

pca<-princomp(letter[, 2:17])
Z<-pca$scores
lambda<-pca$sdev^2
plot(cumsum(lambda)/sum(lambda),type="l",
     xlab="Composantes principales",
     ylab="Proportion de variance expliquée", 
     main="Proportion de variance expliquée selon les composantes principales")

# Mauvaise idée d'appliquer PCA
# Il faudrait voir avec les autres méthodes de sélection de variables
# si on a de meilleurs résultats

# Bike rental

missing_val<-data.frame(apply(bike,2,function(x){sum(is.na(x))}))
names(missing_val)[1]='missing_val'
missing_val

print(unique(bike$yr))
bike$yr <- NULL 

# $instant ne sert à rien, ça correspond simplement aux index

bike$instant <- NULL

# Changement de types
bike$dteday<- as.Date(bike$dteday)
bike$mnth<-as.factor(bike$mnth)
bike$season <- as.factor(bike$season)
bike$holiday<- as.factor(bike$holiday)
bike$weekday<- as.factor(bike$weekday)
bike$workingday<- as.factor(bike$workingday)
bike$weathersit<- as.factor(bike$weathersit)

# Plots 
# https://www.kaggle.com/lakshmi25npathi/bike-rental-count-prediction-using-r

# Plot sur l'année
plot(bike$cnt)
# Plot sur les mois
plot(which(bike$mnth==1), bike$cnt[1:31])
bike_per_month <- bike %>% 
  group_by(mnth) %>% 
  summarise(mnth = mnth[1], cnt=sum(cnt))
bike_per_month$mnth <- as.factor(bike_per_month$mnth)
ggplot(bike_per_month,aes(x=mnth,y=cnt))+theme_bw()+geom_col()+
  labs(x='Mois',y='Locations',title='Comptage par mois')+ 
  geom_bar(stat="identity", fill = "#FF6666")+
  scale_x_discrete(labels= c("Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Sep", "Oct", "Nov", "Dec"))
# Plot sur les saisons
bike_per_season <- bike %>% 
  group_by(season) %>% 
  summarise(season = season[1], cnt=sum(cnt))

ggplot(bike_per_season,aes(x=season,y=cnt))+theme_bw()+geom_col()+
  labs(x='Mois',y='Locations',title='Comptage selon les saisons par mois')+
  scale_x_discrete(limits= c("Printemps", "Ete", "Automne", "Hiver"))+ 
  geom_bar(stat="identity", fill = "#FF6666")
# Plot selon les jours par mois
ggplot(bike,aes(x=mnth,y=cnt,fill=weekday))+theme_bw()+geom_col()+
  labs(x='Saisons',y='Locations',title='Comptage selon les jours par mois')

bike_per_weekday <- bike %>% 
  group_by(weekday) %>% 
  summarise(weekday = weekday[1], cnt=sum(cnt))

ggplot(bike_per_weekday,aes(x=weekday,y=cnt))+theme_bw()+geom_col()+
  labs(x='Mois',y='Locations',title='Comptage selon les saisons par mois')+
  scale_x_discrete(labels= c("Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi","Dimanche"))+ 
  geom_bar(stat="identity", fill = "#FF6666")
# Plot selon les vacances ou non 
ggplot(bike,aes(x=holiday,y=cnt))+geom_col()+theme_bw()+
  labs(x='Vacances',y='Locations',title='Holiday wise distribution of counts')
# Plot selon les vacances, par saison
ggplot(bike,aes(x=holiday,y=cnt, fill=season))+geom_col()+theme_bw()+
  labs(x='Vacances',y='Locations',title='Holiday wise distribution of counts')

# Locations croissantes en printemps et en été et ça redescend nettement à partir d'automne
# En été : locations tous les jours, à peu près pareil. Entre ceux qui partent 
# au travail et d'autres pour les vacances ou autre
# Largement plus de locations pdt les vacances. 
# Après ATTENTION, la plupart des jours sont en non vacances, seuls très peu de jours sont des jours de vacance
which(bike$holiday==1)
# => peut biaiser les résultats

# Outliers
# Humidité 
boxplot(bike$hum,main="Humidité",sub=paste(boxplot.stats(bike$hum)$out))
# index des jours où les humidités sont trop faibles: 
which(bike$hum < 0.25)

# Vitesse du vent
boxplot(bike$windspeed,main="Vitesse du vent",sub=paste(boxplot.stats(bike$windspeed)$out))
# index des jours où la Vitesse du vent est trop élevée: 
which(bike$windspeed > 0.38)

# PCA
pca<-princomp(bike[, 8:11])
Z<-pca$scores
lambda<-pca$sdev^2
plot(cumsum(lambda)/sum(lambda),type="l",
     xlab="Composantes principales",
     ylab="Proportion de variance expliquée", 
     main="Proportion de variance expliquée selon les composantes principales")

# Une variable est inutile, regardons les corrélations

# Correlation
panel.cor <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("R = ", r)
  text(0.5, 0.5, txt, cex = 0.8)
}
# Create the plots
pairs(bike[, 8:11], lower.panel = panel.cor)

# C'est clairement atemp car R[temp, atemp] = 1, enlevons là de notre jeu de données

bike$atemp <- NULL

names(bike)

# En clair, il faudra créer des dummy variables pour s'occuper des variables qualitatives.
# Voir lesquelles on peut regrouper ensemble si c'est utile ou non
# J'ai un doute quant à l'utilité de la variable dteday, vu toutes les autres infos qu'on a

# Régression linéaire pour voir les coefficients significativement non nuls.

model.reg <- lm(cnt ~. , data = bike)
summary(model.reg)

# NA sur dteday et yr. Par contre, des coefficients ont des valeurs étonnament faibles
# tandis que d'autres sont élevées alors que je m'attendais à voir moins.
# Dans tous les cas, la régression linéaire n'est pas très efficace en général donc on utilise d'autres méthodes.

plot(bike$cnt, rstandard(model.reg))
abline(h=0)

# semble suivre une structure croissante = > annonce un mauvais ajustement
# du modèle ou autocorrélation des résidus (non indép, voir TP2).

# Test de normalité

# QQ norm
qqnorm(resid(model.reg))
qqline(resid(model.reg))
# valeurs qui s'éloignent pour les faibles quantiles négatifs

# Test de shapiro-wilk
shapiro.test(resid(model.reg))
# p-value = 0.0009319, il est très faible, donc on peut rejeter l'hypothèse de normalité des résidus
