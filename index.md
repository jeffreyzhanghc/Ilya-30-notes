---
layout: default
---

<!-- Custom CSS -->
<style>
/* Paper cards */
.paper-list {
  display: grid;
  gap: 2rem;
}

.paper-card:hover {
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.paper-card::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  width: 4px;
  background: var(--primary);
}

.paper-meta {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
  color: var(--text-secondary);
  font-size: 0.875rem;
}


/* Tags */
.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.tag {
  background: var(--bg-subtle);
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  color: var(--primary);
  border: 1px solid var(--primary);
}

/* Links */
.paper-links {
  display: flex;
  gap: 1rem;
}

.link-primary, .link-secondary {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.link-primary {
  background: var(--primary);
  color: white;
}

.link-primary:hover {
  background: var(--primary-dark);
}

.link-secondary {
  background: var(--bg-subtle);
  color: var(--text-secondary);
}

.link-secondary:hover {
  background: var(--border);
}
</style>

<div class="container">
    <header class="site-header">
    <h1 class="site-title">Ilya Sutskever's Recommended Reading List Review</h1>
    <p class="site-description">Blog recording some personal thoughts on Ilya's recommended reading, might compare some updated research with the perspective announced during the past, trying to have a foundamental comprehension on current stage AI</p>
    </header>
  <div class="paper-list">
    <article class="paper-card">
      <div class="paper-meta">
        <span>December 16, 2024</span>
        <span>•</span>
        <span>Kaplan et al.</span>
        <span>•</span>
        <span>OpenAI</span>
      </div>
      <h2 class="paper-title">Scaling Laws for Neural Language Models</h2>
      <p class="Summary">
        A review of well-known paper introducing Scaling Law, also including the discussion regarding Test-Time Training and Scaling Law for Downstream Task. 
      </p>
      <div class="tag-list">
        <span class="tag">Scaling Law</span>
        <span class="tag">Test-Time Training</span>
        <span class="tag">Downstream Task</span>
      </div>
      <div class="paper-links">
        <a href="/posts/scaling-laws.html" class="link-primary">
          Read Review
        </a>
        <a href="https://arxiv.org/abs/2001.08361" class="link-secondary" target="_blank" rel="noopener">
          Original Paper
        </a>
      </div>
    </article>

    <!-- Template for future papers -->
    <!-- Copy the article.paper-card structure and update content -->
  </div>
</div>