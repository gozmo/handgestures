find dataset | xargs basename -a | sort | uniq -D | uniq> duplicates
cd dataset/unlabeled ; cat ../duplicates  | xargs rm
