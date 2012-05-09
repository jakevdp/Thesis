import os
import socket

hostname = socket.gethostname()

if hostname == 'pleiades':
    COSMOS_DIR = '/home/jake/COSMOS_DATA/v0.9_public'
elif hostname.endswith('astro.washington.edu'):
    COSMOS_DIR = '/astro/store/student-scratch1/vanderplas/COSMOS/v0.9_public'

BRIGHT_CAT = os.path.join(COSMOS_DIR,'cosmos_bright_cat.asc')
FAINT_CAT  = os.path.join(COSMOS_DIR,'z_cos30/cosmos_faint_cat.asc')

BRIGHT_CAT_NPZ = os.path.join(COSMOS_DIR,'bright_cat.npz')
FAINT_CAT_NPZ = os.path.join(COSMOS_DIR,'faint_cat.npz')

BRIGHT_ZDIST = os.path.join(COSMOS_DIR, 
                            'zdist_bright.asc')

BRIGHT_ZDIST_ZPROB0 = os.path.join(COSMOS_DIR, 
                                   'zdist_bright_zprob0.asc')

FAINT_ZDIST = os.path.join(COSMOS_DIR, 'z_cos30',
                             'cosmos_zdist_faint_w0.asc')

FAINT_ZDIST_W = os.path.join(COSMOS_DIR, 'z_cos30',
                             'cosmos_zdist_faint_w0.asc')

COMBINED_ZDIST = os.path.join(COSMOS_DIR,
                              'zdist_bright_plus_faint_w0.asc')

COMBINED_ZDIST_ZPROB0 = os.path.join(COSMOS_DIR,
                                     'zdist_bright_plus_faint_w0_zprob0.asc')
