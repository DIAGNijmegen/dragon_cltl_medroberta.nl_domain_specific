git init
git add --chmod=+x -- build.sh export.sh train-test.sh
git add -A
git commit -m "Initial release"
git remote add origin https://github.com/DIAGNijmegen/dragon_cltl_medroberta.nl_domain_specific
git push -u origin main
gh repo edit https://github.com/DIAGNijmegen/dragon_cltl_medroberta.nl_domain_specific --description "DRAGON CLTL MedRoBERTa.nl Domain-specific"
