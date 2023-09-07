import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#astar pathfinding algorithm
def astar(start, goal, h_func, neighbors_func,cost_func):
    start,goal = goal,start #im sorry
    open_set = set()
    closed_set = set()
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h_func(start, goal)}

    open_set.add(start)

    while open_set:
        current = None
        current_f_score = None
        for node in open_set:
            if current is None or f_score[node] < current_f_score:
                current = node
                current_f_score = f_score[node]
        paths=[]
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
                paths.append([(x,y) for (x,y) in path])
            path.reverse()
            return paths

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in neighbors_func(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + cost_func(current, neighbor)

            if neighbor not in open_set or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h_func(neighbor, goal)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return None




#import obstacles from an image

#contrast the image
def contrast(file_path,threshold,resx,resy):
 im = Image.open(file_path)
	
 im=im.resize((resx,resy))
 im=np.array(im)
	
 for x in range(im.shape[0]):
  for y in range(im.shape[1]):
   if max(im[x,y][0],im[x,y][1],im[x,y][2])/255 <threshold:
    im[x,y]=[0,0,0,255]
   elif ((im[x,y]-[0,0,255,255])**2).mean(axis=None)<10**2:
    im[x,y]=[0,0,255,255]
   elif ((im[x,y]-[255,0,0,255])**2).mean(axis=None)<10**2:
    im[x,y]=[255,0,0,255]
   else:
    im[x,y]=[255,255,255,255]
 return im

obstacle_image=contrast('media/Untitled130_20230419154340.png',0.5,150,200)







sizex = obstacle_image.shape[0]
sizey = obstacle_image.shape[1]

obstacles=[]#contains points which are obstacles

#loop over all pixels
for x in range(obstacle_image.shape[0]):
 for y in range(obstacle_image.shape[1]):
  
  #make dark pixels obstacles
  if obstacle_image[x,y][0]<255/2:
   obstacles.append((x,y))
  
  #make blue pixels starting point
  if ((obstacle_image[x,y]-[0,0,255,255])**2).mean(axis=None)<100**2:
   start = (x,y)
   
  #make red pixels ending point
  if ((obstacle_image[x,y]-[255,0,0,255])**2).mean(axis=None)<100**2:
   goal = (x,y)
   


# Define the heuristic function
def heuristic(node, goal):
    return np.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

# Define the neighbors function
def neighbors(node):
    x, y = node
    results = []
    if not (x,y) in obstacles:
     if x > 0:
         results.append((x - 1, y))
     if x < sizex - 1:
         results.append((x + 1, y))
     if y > 0:
         results.append((x, y - 1))
     if y < sizey - 1:
         results.append((x, y + 1))
    return results

# Define the cost function
def cost(node1, node2):
    return 1


if not start:
 raise Exception("missing blue start point")
if not goal:
  raise Exception("missing red ending point")

# find path list, all frames
paths_so_many = astar(start, goal, heuristic, neighbors,cost)

#skip some frames to increase speed
paths=[]
for i in range(len(paths_so_many)):
 if i%8==0:
  paths.append(paths_so_many[i])


#multiplication of images in the color sense
#used in plotting in imshow
def im_mul(a,b):
 result=np.zeros(a.shape)
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):
   result[i,j]=np.fmin(a[i][j],b[i][j])
 return result

#plotting

#initial frame
image=np.zeros((sizex,sizey,4))
image[:][:]=[255,255,255,255]

fig,ax = plt.subplots()
ims=[]
ax.set_xticklabels([])
ax.set_yticklabels([])
frame=0


#update animates the path
def update():
 global frame
 for x,y in paths[frame]:
  image[x,y]=[255,0,0,255]
 frame+=1


for i in range(len(paths)):
 update()
 im = plt.imshow(im_mul(image, obstacle_image),vmin=0,vmax=1,animated=True)
 ims.append([im])

ani=animation.ArtistAnimation(fig,ims,interval=10,blit=True)
fig.tight_layout()
plt.show()