import shutil, os

tmpdir = '/tmp/claude-133851/-data-liuj35-quan-fuselage/dc82b4e4-aa2c-4bb9-8ae3-54bec058918f'
tasks  = os.path.join(tmpdir, 'tasks')
scratch = os.path.join(tmpdir, 'scratchpad')

for d in [tasks, scratch]:
    if os.path.isdir(d):
        size = sum(os.path.getsize(os.path.join(r,f)) for r,_,files in os.walk(d) for f in files)
        print(f'{d}: {size/1e6:.1f} MB, files={sum(1 for r,_,fs in os.walk(d) for f in fs)}')
        for entry in os.listdir(d):
            p = os.path.join(d, entry)
            try:
                if os.path.isfile(p): os.remove(p)
                elif os.path.isdir(p): shutil.rmtree(p)
            except Exception as e:
                print(f'  skip {p}: {e}')
        print(f'  cleaned.')
print('done')
