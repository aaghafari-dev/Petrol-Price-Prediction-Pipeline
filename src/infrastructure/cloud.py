"""
Cloud infrastructure integrations (AWS S3 and Kubernetes).
"""
import os
import subprocess
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config.project_config import S3_BUCKET, S3_PREFIX, AWS_REGION, DEPLOYMENT_CONTROL_ENABLED, ROOT


# ============================================================
# AWS S3 Functions
# ============================================================

def _aws():
    """Create AWS S3 client using standard credential chain."""
    import boto3
    return boto3.client('s3', region_name=os.getenv('AWS_REGION', AWS_REGION))


def s3_status():
    """Check S3 bucket configuration and connectivity."""
    bucket = os.getenv('S3_BUCKET', S3_BUCKET)
    prefix = os.getenv('S3_PREFIX', S3_PREFIX).strip('/')
    region = os.getenv('AWS_REGION', AWS_REGION)
    
    out = {
        'configured': bool(bucket),
        'bucket': bucket or None,
        'prefix': prefix,
        'region': region,
        'healthy': False
    }
    
    if not bucket:
        out['message'] = 'S3_BUCKET is not configured. Set it in .env file.'
        return out
    
    try:
        c = _aws()
        c.head_bucket(Bucket=bucket)
        out.update(healthy=True, message='Bucket reachable')
    except Exception as e:
        error_msg = str(e)
        if 'NoSuchBucket' in error_msg:
            out['message'] = 'Bucket does not exist'
        elif 'Forbidden' in error_msg or 'AccessDenied' in error_msg:
            out['message'] = 'Permission denied - check IAM credentials'
        elif 'InvalidAccessKeyId' in error_msg:
            out['message'] = 'Invalid AWS Access Key'
        elif 'ExpiredToken' in error_msg:
            out['message'] = 'AWS credentials expired'
        else:
            out['message'] = 'Connection failed'
        out['error'] = error_msg
    
    return out


def s3_objects(limit=100):
    """List objects in the configured S3 bucket."""
    st = s3_status()
    if not st['healthy']:
        return st | {'objects': []}
    
    try:
        c = _aws()
        r = c.list_objects_v2(
            Bucket=st['bucket'],
            Prefix=st['prefix'] + '/',
            MaxKeys=limit
        )
        return st | {
            'objects': [
                {
                    'key': x['Key'],
                    'size': x['Size'],
                    'last_modified': x['LastModified'].isoformat()
                }
                for x in r.get('Contents', [])
            ]
        }
    except Exception as e:
        return st | {'objects': [], 'list_error': str(e)}


def upload(path, key):
    """Upload a file to S3."""
    st = s3_status()
    if not st['healthy']:
        raise RuntimeError(st.get('error') or st['message'])
    
    full = f"{st['prefix']}/{key}".strip('/')
    _aws().upload_file(str(path), st['bucket'], full)
    return {'uploaded': True, 'bucket': st['bucket'], 'key': full}


# ============================================================
# Kubernetes Functions
# ============================================================

def k8s_cmd(*args, timeout=10):
    """Run a kubectl command."""
    return subprocess.run(
        ['kubectl', *args],
        capture_output=True,
        text=True,
        timeout=timeout
    )


def k8s_status():
    """Get Kubernetes cluster status."""
    out = {
        'available': False,
        'context': None,
        'namespace': None,
        'nodes': 0,
        'ready_nodes': 0,
        'kubectl_version': None
    }
    
    try:
        # Get current context
        ctx = k8s_cmd('config', 'current-context', timeout=5)
        out['context'] = ctx.stdout.strip() if ctx.returncode == 0 else None
        
        # Get namespace
        ns = k8s_cmd('config', 'view', '--minify', '-o', 'jsonpath={..namespace}', timeout=5)
        out['namespace'] = ns.stdout.strip() or 'default'
        
        # Get kubectl version
        ver = k8s_cmd('version', '--client', '-o', 'json', timeout=5)
        if ver.returncode == 0:
            d = json.loads(ver.stdout)
            out['kubectl_version'] = d.get('clientVersion', {}).get('gitVersion')
        
        # Get nodes
        p = k8s_cmd('get', 'nodes', '-o', 'json', timeout=8)
        if p.returncode != 0:
            out['error'] = p.stderr.strip()
            return out
        
        nodes = json.loads(p.stdout).get('items', [])
        out['nodes'] = len(nodes)
        out['ready_nodes'] = sum(
            any(
                c.get('type') == 'Ready' and c.get('status') == 'True'
                for c in n.get('status', {}).get('conditions', [])
            )
            for n in nodes
        )
        out['available'] = True
        out['message'] = 'Cluster reachable'
        
    except subprocess.TimeoutExpired:
        out['error'] = 'kubectl command timed out'
    except Exception as e:
        out['error'] = str(e)
    
    return out


def k8s_resources(kind='pods', namespace=''):
    """Get Kubernetes resources (pods, deployments, etc.)."""
    args = ['get', kind, '-o', 'json']
    if namespace:
        args += ['-n', namespace]
    
    p = k8s_cmd(*args, timeout=12)
    if p.returncode:
        return {'ok': False, 'error': p.stderr.strip(), 'items': []}
    
    d = json.loads(p.stdout)
    items = []
    for x in d.get('items', []):
        items.append({
            'name': x['metadata'].get('name'),
            'namespace': x['metadata'].get('namespace'),
            'status': x.get('status', {}).get(
                'phase',
                x.get('status', {}).get('conditions', [{}])[0].get('status', '')
            )
        })
    
    return {'ok': True, 'items': items}


def apply_k8s(confirm=False):
    """Apply Kubernetes manifest (requires confirmation)."""
    enabled = os.getenv('DEPLOYMENT_CONTROL_ENABLED', 'false').lower() == 'true'
    
    if not (confirm and enabled):
        raise RuntimeError(
            'Deployment is disabled. '
            'Set DEPLOYMENT_CONTROL_ENABLED=true and confirm DEPLOY.'
        )
    
    p = k8s_cmd('apply', '-f', str(ROOT / 'k8s'), timeout=90)
    return {'ok': p.returncode == 0, 'stdout': p.stdout, 'stderr': p.stderr}