/**
 * ARM64 Sequences — HIR opcode dispatch for ARM64
 */
#ifndef XENIA_CPU_BACKEND_A64_A64_SEQUENCES_H_
#define XENIA_CPU_BACKEND_A64_A64_SEQUENCES_H_

#include <unordered_map>
#include "xenia/base/logging.h"
#include "xenia/cpu/hir/instr.h"

namespace xe {
namespace cpu {
namespace backend {
namespace a64 {

class A64Emitter;

typedef bool (*SequenceSelectFn)(A64Emitter&, const hir::Instr*);

extern std::unordered_map<uint32_t, SequenceSelectFn> sequence_table;

bool SelectSequence(A64Emitter* e, const hir::Instr* i,
                    const hir::Instr** new_tail);

void RegisterSequences();

}  // namespace a64
}  // namespace backend
}  // namespace cpu
}  // namespace xe

#endif  // XENIA_CPU_BACKEND_A64_A64_SEQUENCES_H_
