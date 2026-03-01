/**
 * vLLM Ascend Documentation - Enhanced JavaScript Interactions
 */

(function() {
  'use strict';

  function initScrollSpy() {
    const sections = document.querySelectorAll('.section[id], [id]');
    const tocLinks = document.querySelectorAll('.toc-href, .bd-toc a');
    if (!sections.length || !tocLinks.length) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (!entry.isIntersecting) return;
        tocLinks.forEach(link => {
          link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`);
        });
      });
    }, { rootMargin: '-20% 0px -70% 0px' });

    sections.forEach(section => observer.observe(section));
  }

  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;
        const target = document.querySelector(href);
        if (!target) return;

        e.preventDefault();
        const offset = target.getBoundingClientRect().top + window.pageYOffset - 100;
        window.scrollTo({ top: offset, behavior: 'smooth' });
        history.pushState(null, null, href);
      });
    });
  }

  function initCodeBadges() {
    document.querySelectorAll('.highlight').forEach(block => {
      const lang = block.className.split(' ').find(c => c.startsWith('highlight-'))?.replace('highlight-', '');
      if (lang && lang !== 'text') block.setAttribute('data-lang', lang);
    });
  }

  function initThemeTransition() {
    document.querySelector('#theme-switch, .theme-switch')?.addEventListener('click', () => {
      document.body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
      setTimeout(() => document.body.style.transition = '', 300);
    });
  }

  function init() {
    const initAll = () => setTimeout(() => {
      initScrollSpy();
      initSmoothScroll();
      initCodeBadges();
      initThemeTransition();
    }, 200);

    document.readyState === 'loading'
      ? document.addEventListener('DOMContentLoaded', initAll)
      : initAll();
  }

  init();
})();
