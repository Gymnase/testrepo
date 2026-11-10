# hn-tech-digest

Workflow Mistral exécuté à la demande : lit la page d'accueil de HackerNews, ne garde que les
actualités du secteur tech, produit des résumés en français, génère un audio podcast de 1 à 2
minutes (TTS Voxtral), puis envoie le tout par email Gmail avec le MP3 en pièce jointe.

## Contenu

- `worker/workflows/hn_digest.py` : le workflow `hn-tech-digest` et ses activités
  (fetch HN, curation + résumés par LLM, génération audio, envoi Gmail).
- `Dockerfile.worker` : image du worker pour le déploiement managé Mistral Cloud.
- `tests/` : tests locaux (worker de test Temporal, effets externes mockés).

## Entrées du workflow (`HnDigestInput`)

| Champ | Défaut | Description |
|---|---|---|
| `recipient_email` | `michel@puharre.com` | Destinataire (votre boîte Gmail) |
| `max_stories` | `10` | Nombre max d'actualités tech retenues |
| `language` | `fr` | Langue des résumés et du podcast |
| `dry_run` | `false` | `true` : prépare le digest sans envoyer l'email |

Le workflow s'exécute `on_behalf_of` : les identifiants Gmail utilisés sont ceux de
l'utilisateur qui déclenche l'exécution (connecteur `gmail`).

## Lancer localement

```bash
uv sync
uv run pytest
cp .env.example .env   # puis renseigner MISTRAL_API_KEY
uv run python -m worker.entrypoints.worker
```

## Déclencher une exécution

Le workflow n'est PAS planifié : il démarre quand vous l'exécutez.

```bash
curl -X POST https://api.mistral.ai/v1/workflows/hn-tech-digest/execute \
  -H "Authorization: Bearer $MISTRAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {}}'
```

Ou depuis le Studio : Workflows > `hn-tech-digest` > Start workflow.

## Déploiement managé

Le déploiement Mistral Cloud construit l'image depuis ce dépôt GitHub
(`Dockerfile.worker`, worker authentifié par token de service account injecté).
