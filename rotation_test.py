import rs_snapshot_rotation
from multiprocessing import Process, Queue
q0 = Queue()
p0 = Process(target=rs_snapshot_rotation.rotation_only, args=(q0,))
q0.put('snapshot_1619001238.6786714.ply')
p0.start()
p0.join()

