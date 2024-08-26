from mpi4py import MPI
import socket
import os
import sys
import subprocess
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(rank)
if rank == 0:
  data = socket.gethostbyname(socket.gethostname())
  os.system('ray start --head --port=6379 > _dump.log')
  with open ('_dump.log', 'rt') as myfile:
    for line in myfile:
      ip = line.rsplit(' ')
      data = ip[-1].rstrip()
      break
else:
  data = None

data = comm.bcast(data, root=0)
data = data + ":6379"
cmd = "ray start --address={0} --redis-password='5241590000000000'".format(data)
if rank != 0:
  os.system(cmd)

comm.Barrier()
if rank == 0:
  os.system('python Main.py --data-name=Ecoli')
