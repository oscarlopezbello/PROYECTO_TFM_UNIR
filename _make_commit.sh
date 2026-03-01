#!/bin/bash
TREE=$(git write-tree)
COMMIT=$(echo "carga de proyecto tfm master en inteligencia artificial UNIR" | git commit-tree "$TREE")
git update-ref refs/heads/main "$COMMIT"
echo "Done: $COMMIT"
