import { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { ReconstructionResult } from "../lib/api";

/** Colour ramp for the point cloud: the achromatic page palette plus a single
 *  warm signal tone, so depth reads without introducing a second hue. */
const CLOUD_NEAR = new THREE.Color("#f7f7f7");
const CLOUD_FAR = new THREE.Color("#5c5c5c");
const TRAJ_OPTIMISED = "#fafafa";
const TRAJ_ODOMETRY = "#d08a2c";

interface SceneProps {
  result: ReconstructionResult;
  showOdometry: boolean;
  showCloud: boolean;
  showKeyframes: boolean;
  pointSize: number;
}

/** Sparse landmarks, drawn as a single buffer so 60k points stay cheap. */
function PointCloud({ result, pointSize }: { result: ReconstructionResult; pointSize: number }) {
  const geometry = useMemo(() => {
    const positions = new Float32Array(result.cloud.positions);
    const source = result.cloud.colors;
    const colors = new Float32Array(positions.length);

    // Blend each landmark's sampled colour toward a depth-derived tone. Raw
    // sampled colour alone is noisy on sparse points; pure depth colour loses
    // the scene. The mix keeps structure readable in a dark scene.
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 1; i < positions.length; i += 3) {
      if (positions[i] < minY) minY = positions[i];
      if (positions[i] > maxY) maxY = positions[i];
    }
    const span = maxY - minY || 1;
    const tone = new THREE.Color();

    for (let i = 0, p = 0; i < positions.length; i += 3, p += 3) {
      const t = (positions[i + 1] - minY) / span;
      tone.copy(CLOUD_FAR).lerp(CLOUD_NEAR, t);
      const r = (source[p] ?? 200) / 255;
      const g = (source[p + 1] ?? 200) / 255;
      const b = (source[p + 2] ?? 200) / 255;
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      colors[i] = tone.r * (0.45 + 0.55 * luma);
      colors[i + 1] = tone.g * (0.45 + 0.55 * luma);
      colors[i + 2] = tone.b * (0.45 + 0.55 * luma);
    }

    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geom.computeBoundingSphere();
    return geom;
  }, [result]);

  return (
    <points geometry={geometry}>
      <pointsMaterial
        size={pointSize}
        vertexColors
        sizeAttenuation
        transparent
        opacity={0.95}
        depthWrite={false}
      />
    </points>
  );
}

function Path({ positions, color, width }: { positions: number[]; color: string; width: number }) {
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
    return geom;
  }, [positions]);
  if (positions.length < 6) return null;
  return (
    <line>
      <primitive object={geometry} attach="geometry" />
      <lineBasicMaterial color={color} linewidth={width} />
    </line>
  );
}

/** Camera frustums at keyframes, showing where and which way the camera looked. */
function Keyframes({ result, scale }: { result: ReconstructionResult; scale: number }) {
  const geometry = useMemo(() => {
    const verts: number[] = [];
    const s = scale;
    // unit frustum in camera space: apex at origin, rectangle one unit ahead
    const corners = [
      [-s * 0.6, -s * 0.4, s],
      [s * 0.6, -s * 0.4, s],
      [s * 0.6, s * 0.4, s],
      [-s * 0.6, s * 0.4, s],
    ];
    for (const kf of result.keyframes) {
      const R = kf.rotation;
      const t = kf.position;
      const toWorld = (v: number[]) => [
        R[0] * v[0] + R[1] * v[1] + R[2] * v[2] + t[0],
        R[3] * v[0] + R[4] * v[1] + R[5] * v[2] + t[1],
        R[6] * v[0] + R[7] * v[1] + R[8] * v[2] + t[2],
      ];
      const apex = t;
      const world = corners.map(toWorld);
      for (const c of world) verts.push(apex[0], apex[1], apex[2], c[0], c[1], c[2]);
      for (let i = 0; i < 4; i++) {
        const a = world[i];
        const b = world[(i + 1) % 4];
        verts.push(a[0], a[1], a[2], b[0], b[1], b[2]);
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
    return geom;
  }, [result, scale]);

  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial color="#8a8a8a" transparent opacity={0.5} />
    </lineSegments>
  );
}

/** Bounding box of the reconstruction, trimmed so a few stray landmarks cannot
 *  pull the framing far out. Used for both the initial camera placement and the
 *  orbit target: leaving OrbitControls pointed at the origin while the camera
 *  looks elsewhere is what pushes the scene off to one side. */
function useFraming(result: ReconstructionResult) {
  return useMemo(() => {
    const traj = result.trajectory.positions;
    const cloud = result.cloud.positions;
    const box = new THREE.Box3();
    const v = new THREE.Vector3();

    for (let i = 0; i < traj.length; i += 3) {
      box.expandByPoint(v.set(traj[i], traj[i + 1], traj[i + 2]));
    }

    if (cloud.length >= 3) {
      const centre = new THREE.Vector3();
      const n = cloud.length / 3;
      for (let i = 0; i < cloud.length; i += 3) {
        centre.add(v.set(cloud[i], cloud[i + 1], cloud[i + 2]));
      }
      centre.divideScalar(n);

      const dists: number[] = [];
      for (let i = 0; i < cloud.length; i += 3) {
        dists.push(v.set(cloud[i], cloud[i + 1], cloud[i + 2]).distanceTo(centre));
      }
      dists.sort((a, b) => a - b);
      const cutoff = dists[Math.floor(dists.length * 0.95)] ?? Infinity;
      for (let i = 0; i < cloud.length; i += 3) {
        if (v.set(cloud[i], cloud[i + 1], cloud[i + 2]).distanceTo(centre) <= cutoff) {
          box.expandByPoint(v);
        }
      }
    }

    if (box.isEmpty()) {
      return { centre: new THREE.Vector3(), radius: 1 };
    }
    return {
      centre: box.getCenter(new THREE.Vector3()),
      radius: Math.max(box.getSize(new THREE.Vector3()).length() * 0.5, 0.5),
    };
  }, [result]);
}

function AutoFrame({ centre, radius }: { centre: THREE.Vector3; radius: number }) {
  const { camera } = useThree();
  const applied = useRef(false);

  useFrame(() => {
    if (applied.current) return;
    applied.current = true;
    camera.position.set(
      centre.x + radius * 1.5,
      centre.y + radius * 1.0,
      centre.z + radius * 1.5,
    );
    camera.lookAt(centre);
    camera.near = radius / 500;
    camera.far = radius * 60;
    camera.updateProjectionMatrix();
  });
  return null;
}

export function Scene({
  result,
  showOdometry,
  showCloud,
  showKeyframes,
  pointSize,
}: SceneProps) {
  const { centre, radius } = useFraming(result);
  const frustumScale = Math.max(radius * 0.022, 0.01);

  return (
    <Canvas
      camera={{ fov: 55, near: 0.01, far: 1000, position: [3, 2, 3] }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: false }}
      onCreated={({ gl }) => gl.setClearColor("#141414")}
    >
      <AutoFrame centre={centre} radius={radius} />
      <ambientLight intensity={0.8} />

      {showCloud && <PointCloud result={result} pointSize={pointSize} />}
      {showKeyframes && <Keyframes result={result} scale={frustumScale} />}

      {showOdometry && (
        <Path positions={result.odometry_trajectory.positions} color={TRAJ_ODOMETRY} width={1} />
      )}
      <Path positions={result.trajectory.positions} color={TRAJ_OPTIMISED} width={2} />

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.7}
        target={centre}
      />
    </Canvas>
  );
}
