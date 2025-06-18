import os
import math
import dxfgrabber
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def draw_block(block, out_file):
    fig, ax = plt.subplots()
    for e in block._entities:
        if e.dxftype == 'LINE':
            xs = [e.start[0], e.end[0]]
            ys = [e.start[1], e.end[1]]
            ax.plot(xs, ys, color='black')
        elif e.dxftype in ('POLYLINE', 'LWPOLYLINE'):
            points = e.points
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if getattr(e, 'is_closed', False):
                xs.append(points[0][0])
                ys.append(points[0][1])
            ax.plot(xs, ys, color='black')
        elif e.dxftype == 'ARC':
            # approximate arc using many line segments
            start_angle = math.radians(e.start_angle)
            end_angle = math.radians(e.end_angle)
            if end_angle < start_angle:
                end_angle += 2*math.pi
            angles = [start_angle + (end_angle-start_angle)*i/30 for i in range(31)]
            xs = [e.center[0] + e.radius*math.cos(a) for a in angles]
            ys = [e.center[1] + e.radius*math.sin(a) for a in angles]
            ax.plot(xs, ys, color='black')
        elif e.dxftype == 'CIRCLE':
            angle = [2*math.pi*i/60 for i in range(61)]
            xs = [e.center[0] + e.radius*math.cos(a) for a in angle]
            ys = [e.center[1] + e.radius*math.sin(a) for a in angle]
            ax.plot(xs, ys, color='black')
        elif e.dxftype == 'TEXT':
            ax.text(e.insert[0], e.insert[1], e.text, fontsize=8)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)

def safe_name(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)

def process_file(path, out_dir):
    doc = dxfgrabber.readfile(path)
    for block in doc.blocks:
        if block.name.startswith('*'):
            continue
        out_name = f"{safe_name(os.path.splitext(os.path.basename(path))[0])}_{safe_name(block.name)}.png"
        out_path = os.path.join(out_dir, out_name)
        draw_block(block, out_path)
        print('Saved', out_path)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize DXF blocks as images')
    parser.add_argument('input_dir', help='directory with dxf files')
    parser.add_argument('output_dir', help='directory for png files')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith('.dxf'):
            process_file(os.path.join(args.input_dir, fname), args.output_dir)

if __name__ == '__main__':
    main()
