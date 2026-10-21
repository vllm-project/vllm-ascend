(function () {
  function filterTabbedToc() {
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
  }

  document$.subscribe(() => {
    filterTabbedToc()
    revealTabbedTarget()
  })
  window.addEventListener('hashchange', revealTabbedTarget)
})()
