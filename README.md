# ChessEnigne

## Search
- [x] Negamax
- [x] Alpha-beta pruning
- [ ] Iterative deepening
- [ ] Principal Variation Search (PVS)
- [ ] Aspiration windows
- [x] Quiescence search

## Move ordering
- [x] Hash move
- [x] MVV-LVA
- [x] SEE
- [ ] Killer heuristic
- [ ] History heuristic
- [ ] Countermove heuristic

## Transposition table
- [x] TT probing
- [x] TT storing
- [x] Exact / lower / upper bounds
- [ ] Replacement policy
- [x] Hash move extraction

## Pruning and reductions
- [ ] Null move pruning
- [ ] Late Move Reductions (LMR)
- [ ] Futility pruning
- [ ] Reverse futility pruning
- [ ] Razoring
- [ ] Check extensions
- [ ] Singular extensions

## Evaluation
- [ ] NNUE inference
- [ ] NNUE accumulator updates
- [ ] Phase / scaling logic
- [ ] Drawish / endgame scaling

## Time management
- [ ] Basic time allocation
- [ ] Soft / hard time limits
- [ ] Stop conditions
- [ ] PV instability handling

## UCI / engine behavior
- [x] UCI loop
- [x] Position parsing
- [ ] Go / stop handling
- [ ] Search info output
- [ ] Bench / debug commands

## Testing
- [x] Perft validation
- [x] TT correctness tests
- [ ] TT hit rate tests
- [ ] TT collision rate tests
- [ ] SEE correctness tests
- [ ] Search regression tests
- [ ] Self-play testing
- [ ] SPRT / Elo testing
more tests to be added
