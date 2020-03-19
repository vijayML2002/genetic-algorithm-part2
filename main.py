#importing the req lib

%tensorflow_version 2.x
import tensorflow as tf

import numpy as np
from keras.callbacks import History


#creating a simple dataset

no_game = 1000
def create():
  data1 = []
  data2 = []
  for x in range(no_game):
    y = x+10
    data1.append(x)
    data2.append(y)

  print('DATA CREATED !!!')
  return data1,data2

x,y = create()
x = np.array(x).reshape(len(x),1,1)
y = np.array(y).reshape(len(x),1)


def create_person(param1,param2,param3,param4):
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Input(shape=(1,1)))
  for layer in range(param1): #param1 - number of layer 
    model.add(tf.keras.layers.Dense(param2,activation=param3)) #param2 - middle layer, #param3 - activation fun
  model.add(tf.keras.layers.Dense(1))
  model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(param4))
  return model

#defing our parameters list

parameter_1 = [1,2,3,4,5,6]
parameter_2 = [4,16,32,64,128,256]
parameter_3 = ['relu','tanh','sigmoid']
parameter_4 = [0.5,0.1,0.05,0.01,0.005,0.001]

#checking the model with sample parameters
param1 = np.random.choice(parameter_1)
param2 = np.random.choice(parameter_2)
param3 = np.random.choice(parameter_3)
param4 = np.random.choice(parameter_4)

pers = create_person(param1,param2,param3,param4)

#sum of the model

pers.summary()

#def the random function for selecting 

def get_random_param():
  param1 = np.random.choice(parameter_1)
  param2 = np.random.choice(parameter_2)
  param3 = np.random.choice(parameter_3)
  param4 = np.random.choice(parameter_4)

  return param1,param2,param3,param4

#creating the fitness function

def check_fitness(person):
  history = History()
  person.fit(x,y,epochs=1,verbose=0,callbacks=[history])
  losses = history.history['loss']
  losses = losses[len(losses)-1]
  fit_score = 1/losses
  
  return fit_score


#intializing the random population

def create_population(num):
  no_mem = num

  population = []
  population_attributes = []

  for per in range(no_mem):
    param1,param2,param3,param4 = get_random_param()
    person = create_person(param1,param2,param3,param4)
    population.append(person)
    population_attributes.append([param1,param2,param3,param4])

  print('CREATED POPULATION !!!')
  return population,population_attributes


#def the selection

def selection(people,atts,basic_fit):
  selected = []
  selected_att = []

  for person,att in zip(people,atts):
    fit_score = check_fitness(person)
    if fit_score>basic_fit:
      selected.append(person)
      selected_att.append(att)

  #print('PERSON SELECTED - {}'.format(len(selected)))
  return selected,selected_att


#def for the pairing

def pairing(people,atts):
  paired = []
  paired_att = []

  for i in range(len(people)-1):
    mate1 = people[i]
    mate2 = people[i+1]

    mate1_att = atts[i]
    mate2_att = atts[i+1]

    mate = [mate1,mate2]
    mate_att = [mate1_att,mate2_att]
    
    paired.append(mate)
    paired_att.append(mate_att)

  #print('ALL PEOPLE PAIRED !!!')
  return paired,paired_att

#def the crossover
off_prob = 0

def crossover(mates,atts):
  offspring = []
  offspring_att = []

  if np.random.uniform(0,1)>off_prob:
    for mate,att in zip(mates,atts):
      mate1 = mate[0]
      mate2 = mate[1]

      mate1_att = att[0]
      mate2_att = att[1]

      prob = np.random.randint(0,3)
      
      gene1 = mate1_att[prob]
      gene2 = mate2_att[prob]

      mate2_att[prob] = gene1
      mate1_att[prob] = gene2

      offspring1_att = mate1_att
      offspring2_att = mate2_att

      off1_param1,off1_param2,off1_param3,off1_param4 = offspring1_att[0],offspring1_att[1],offspring1_att[2],offspring1_att[3]
      off2_param1,off2_param2,off2_param3,off2_param4 = offspring2_att[0],offspring2_att[1],offspring2_att[2],offspring2_att[3]

      offspring1 = create_person(off1_param1,off1_param2,off1_param3,off1_param4)
      offspring2 = create_person(off2_param1,off2_param2,off2_param3,off2_param4)

      offspring.append(offspring1)
      offspring.append(offspring2)

      offspring_att.append(offspring1_att)
      offspring_att.append(offspring2_att)

  #print('CROSS OVER COMPLETED - {}'.format(len(offspring)))
  return offspring,offspring_att


#def the mutation

def mutate(people,atts):

  if np.random.uniform(0,1)>0.92:
    index_att = np.random.randint(0,len(people))

    person = people[index_att]
    person_att = atts[index_att]

    param1,param2,param3,param4 = get_random_param()
    param = [param1,param2,param3,param4]
    
    ind = np.random.randint(0,3)

    person_att[ind] = param[ind]
    p1_param1,p1_param2,p1_param3,p1_param4 = person_att[0],person_att[1],person_att[2],person_att[3]

    new_mut = create_person(p1_param1,p1_param2,p1_param3,p1_param4)

    people[index_att] = new_mut
    atts[index_att] = person_att

    #print('MUTATION TAKEN PLACED')

  #else:
    #print('MUTATION DOES NOT TAKES PLACE')

  return people,atts


#def for checkpoint of this evolution

def check_point(people):
  
  for person in people:
    fit_score = check_fitness(person)
    if fit_score>10:
      print('EVOLUTION COMPLETED !!!!')
      return 1
  
  return 0

#def the real evolution of nn

def evolve():
  generation_no = 0
  basic_fit = 1e-5
  population,population_attributes = create_population(50)
  while True:
    print('GENERATION - {}  ALIVE - {}'.format(generation_no,len(population)))
    selected , selected_att = selection(population,population_attributes,basic_fit)
    if len(selected)==0:
      print('EVOLUTION FAILED')
      break
    paired , paired_att = pairing(selected,selected_att)
    crossed , crossed_att = crossover(paired,paired_att)
    mutated , mutated_att = mutate(crossed,crossed_att)
    if check_point(mutated):
      print('EVOLUTION COMPLETED')
      break
    population = mutated
    population_attributes = mutated_att
    generation_no += 1
    if generation_no%3==0:
      basic_fit *= 100


evolve() #this is the evolution function
















  
