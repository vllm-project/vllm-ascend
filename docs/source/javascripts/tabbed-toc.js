(function () {
  function hideTabbedToc() {
    const title = document.querySelector('.md-content h1')
    if (!title || !/^GLM-5\.(2|3-Flash)(?:\s|$)/.test(title.textContent.trim())) {
      return
    }

    // Keep tab contents in the page without repeating them in the navigation.
    document.querySelectorAll('.md-nav--secondary a[href]').forEach(link => {
      const target = document.getElementById(decodeURIComponent(link.hash.slice(1)))
      if (target && target.closest('.tabbed-block')) {
        link.closest('.md-nav__item').remove()
      }
    })
  }

  function revealTabbedTarget() {
    if (!window.location.hash) {
      return
    }

    const id = decodeURIComponent(window.location.hash.slice(1))
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
    hideTabbedToc()
    revealTabbedTarget()
  })
  window.addEventListener('hashchange', revealTabbedTarget)
})()
