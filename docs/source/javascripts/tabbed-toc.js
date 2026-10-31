(function () {
  function setTocExpanded(item, expanded) {
    const button = item.querySelector(':scope > .glm-toc-toggle')
    const children = item.querySelector(':scope > nav')
    if (!button || !children) {
      return
    }

    button.setAttribute('aria-expanded', String(expanded))
    children.hidden = !expanded
    const chinese = document.documentElement.lang.startsWith('zh')
    const action = chinese
      ? (expanded ? '收起子章节' : '展开子章节')
      : (expanded ? 'Collapse subsections' : 'Expand subsections')
    button.setAttribute('aria-label', `${action}: ${children.getAttribute('aria-label')}`)
  }

  function expandTocTarget() {
    document.querySelectorAll('.md-nav--secondary a[href]').forEach(link => {
      if (link.hash !== window.location.hash || !link.hash) {
        return
      }

      let item = link.closest('.md-nav__item')
      while (item && item.closest('.md-nav--secondary')) {
        setTocExpanded(item, true)
        item = item.parentElement.closest('.md-nav__item')
      }
    })
  }

  function initCollapsibleToc() {
    const content = document.querySelector('.md-content')
    if (!content || !content.querySelector('[id^="glm52-53-"], [id^="glm53-flash-"]')) {
      return
    }

    // Keep tab contents in the page without repeating them in the navigation.
    document.querySelectorAll('.md-nav--secondary a[href]').forEach(link => {
      const hash = link.hash.slice(1)
      if (!hash) {
        return
      }

      let id
      try {
        id = decodeURIComponent(hash)
      } catch (error) {
        return
      }
      const target = document.getElementById(id)
      if (target && target.closest('.tabbed-block')) {
        link.closest('.md-nav__item')?.remove()
      }
    })

    document.querySelectorAll('.md-nav--secondary .md-nav__item').forEach((item, index) => {
      const link = item.querySelector(':scope > a.md-nav__link')
      const children = item.querySelector(':scope > nav')
      const hasChildren = children && children.querySelector(':scope > ul > li > a.md-nav__link')
      if (!link || !hasChildren || item.querySelector(':scope > .glm-toc-toggle')) {
        return
      }

      item.classList.add('glm-toc-branch')
      children.id = children.id || `glm-toc-children-${index}`
      const button = document.createElement('button')
      button.type = 'button'
      button.className = 'glm-toc-toggle'
      button.setAttribute('aria-controls', children.id)
      const arrow = document.createElement('span')
      arrow.setAttribute('aria-hidden', 'true')
      arrow.textContent = '›'
      button.appendChild(arrow)
      link.after(button)
      setTocExpanded(item, false)
      button.addEventListener('click', () => {
        setTocExpanded(item, button.getAttribute('aria-expanded') !== 'true')
      })
      link.addEventListener('click', () => setTocExpanded(item, true))
    })
    expandTocTarget()
  }

  function revealTabbedTarget() {
    if (!window.location.hash) {
      return
    }

    let id
    try {
      id = decodeURIComponent(window.location.hash.slice(1))
    } catch (error) {
      return
    }
    const target = document.getElementById(id)
    if (!target) {
      return
    }

    let block = target.closest('.tabbed-block')
    while (block) {
      const content = block.parentElement
      const tabSet = content && content.parentElement
      if (!tabSet || !tabSet.classList.contains('tabbed-set')) {
        break
      }

      const blocks = Array.from(
        tabSet.querySelectorAll(':scope > .tabbed-content > .tabbed-block')
      )
      const index = blocks.indexOf(block)
      const inputs = tabSet.querySelectorAll(':scope > input')
      if (index >= 0 && inputs[index] && !inputs[index].checked) {
        inputs[index].click()
      }

      block = tabSet.closest('.tabbed-block')
    }

    window.requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start' })
    })
    expandTocTarget()
  }

  document$.subscribe(() => {
    initCollapsibleToc()
    revealTabbedTarget()
  })
  window.addEventListener('hashchange', revealTabbedTarget)
})()
