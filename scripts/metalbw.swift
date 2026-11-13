// metalbw.swift - the GPU-side streaming number for the Metal path, matching scripts/membw.c.
//
// The CPU-side benchmark answers "how fast can the host stream bytes". A Metal backend does not run
// there: its kernels read weights through the GPU's own path to the same unified memory, and on Apple
// silicon the GPU gets much closer to the specification than the CPU cluster does.
//
// Build and run:
//   swiftc -O -o metalbw scripts/metalbw.swift
//   ./metalbw 1024            # 1024 MiB, prints read and copy GB/s, five passes, best reported
import Foundation
import Metal

let mib = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 1024
let n = mib * 1024 * 1024 / MemoryLayout<Float>.size

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }

let src = """
#include <metal_stdlib>
using namespace metal;
kernel void copy_k(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid];
}
kernel void read_k(device const float *in [[buffer(0)]], device float *sink [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    float v = in[gid];
    if (v == 1234.5678f) { sink[0] = v; }
}
"""
guard let library = try? device.makeLibrary(source: src, options: nil) else { fatalError("library") }
let copyFn = library.makeFunction(name: "copy_k")!
let readFn = library.makeFunction(name: "read_k")!
let copyPso = try! device.makeComputePipelineState(function: copyFn)
let readPso = try! device.makeComputePipelineState(function: readFn)
let queue = device.makeCommandQueue()!

func buffer(_ bytes: Int) -> MTLBuffer {
    device.makeBuffer(length: bytes, options: .storageModeShared)!
}
let a = buffer(n * 4)
let b = buffer(n * 4)
let sink = buffer(4)
memset(a.contents(), 1, n * 4)

let threadsPerGroup = 256
let groups = (n + threadsPerGroup - 1) / threadsPerGroup
let count = MTLSize(width: groups, height: 1, depth: 1)
let size = MTLSize(width: threadsPerGroup, height: 1, depth: 1)

func time(_ pso: MTLComputePipelineState, _ args: [MTLBuffer], movedBytes: Double) -> Double {
    var best = Double.infinity
    for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        for (i, buf) in args.enumerated() { enc.setBuffer(buf, offset: 0, index: i) }
        enc.dispatchThreadgroups(count, threadsPerThreadgroup: size)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let seconds = cb.gpuEndTime - cb.gpuStartTime
        if seconds > 0 { best = min(best, movedBytes / seconds / 1e9) }
    }
    return best
}

let readGbs = time(readPso, [a, sink], movedBytes: Double(n * 4))
let copyGbs = time(copyPso, [a, b], movedBytes: Double(n * 8))
print("metalbw \(mib) MiB on \(device.name): read \(String(format: "%.1f", readGbs)) GB/s, "
      + "copy \(String(format: "%.1f", copyGbs)) GB/s")
